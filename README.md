---
layout: post
title: Q Öğrenme (Q Learning)
author: Bahri ABACI
categories:
- Makine Öğrenmesi
- Nümerik Yöntemler
- Veri Analizi
thumbnail: /assets/post_resources/reinforcement_learning_q_learning/thumbnailq.png
---
İlk olarak Christopher JCH Watkins and Peter Dayan tarafından 1992 yılında literatüre kazandırılan Q Öğrenme (Q Learning) yöntemi, 2013 yılında [DeepMind](https://deepmind.com) yapay zeka şirketinin kurucuları tarafından yayınlanan *"Playing atari with deep reinforcement learning"* makalesi ile oldukça popüler hale gelen bir pekiştirmeli öğrenme yöntemidir. Algoritmanın matematiksel temellerinin oluşturulduğu [Pekiştirmeli Öğrenme]({% post_url 2020-01-05-pekistirmeli-ogrenme-reinforcement-learning %}) yazımızda da bahsedildiği gibi pekiştirmeli öğrenme, [K-Means]({% post_url 2015-08-28-k-means-kumeleme-algoritmasi %}), [Temel Bileşen Analizi]({% post_url 2019-09-01-temel-bilesen-analizi-principal %}), [Karar Ağaçları]({% post_url 2015-11-01-karar-agaclari-decision-trees %}), [Lojistik Regresyon Analizi]({% post_url 2015-07-23-lojistik-regresyon-analizi %}) ve Destek Vektör Makineleri gibi yöntemlerin hepsinden farklı olarak herhangi bir veriye ihtiyaç duymadan öğrenme yapabilmektedir. Yöntem verilen kurallar çerçevesinde durum uzayını (state-space) keşfederek her durum için elde edeceği ödülü en büyüklemesini sağlayan hareketi (action) öğrenmeye çalışmaktadır.

<!--more-->

Bu yazımızda [Pekiştirmeli Öğrenme]({% post_url 2020-01-05-pekistirmeli-ogrenme-reinforcement-learning %}) yazımızda türetilen Q öğrenme yönteminin matematiksel ifadesi kullanılarak Q öğrenme algoritması incelenecek ve kodlanacaktır. Eğer pekiştirmeli öğrenme mantığı veya burada kullanılacak olan Bellman denklemi, kalite fonksiyonu gibi denklemler hakkında eksiğiniz olduğunu düşünüyorsanız öncelikle [bir önceki yazıyı]({% post_url 2020-01-05-pekistirmeli-ogrenme-reinforcement-learning %}) okumanızı tavsiye ederim.

Öğrenme işleminin matematiksel olarak nasıl yapıldığına geçmeden önce, kullanacağımız matematiksel sembolleri ve anlamlarını bir tablo şeklinde yazalım.

| Sembol | Açıklama |
:-------------------------:|:-------------------------
| !["s_t"](https://render.githubusercontent.com/render/math?math=s_t) | !["t"](https://render.githubusercontent.com/render/math?math=t) anındaki durum (state) |
| !["a_t \in A"](https://render.githubusercontent.com/render/math?math=a_t%20%5cin%20A) | !["A"](https://render.githubusercontent.com/render/math?math=A) hareketler uzayından alınan bir hareket (action) |
| !["s^\prime_t=s_{t+1}"](https://render.githubusercontent.com/render/math?math=s%5e%5cprime_t%3ds_%7bt%2b1%7d) | !["t"](https://render.githubusercontent.com/render/math?math=t) anında !["a"](https://render.githubusercontent.com/render/math?math=a) hareketi ile geçilen yeni durum (new state) |
| !["r_t = r(s_t, a_t, s^{\prime}_t)"](https://render.githubusercontent.com/render/math?math=r_t%20%3d%20r%28s_t%2c%20a_t%2c%20s%5e%7b%5cprime%7d_t%29) | !["s_t"](https://render.githubusercontent.com/render/math?math=s_t) durumunda !["a_t"](https://render.githubusercontent.com/render/math?math=a_t) kararı ile !["s^{\prime}_t"](https://render.githubusercontent.com/render/math?math=s%5e%7b%5cprime%7d_t) durumuna geçilerek elde edilen ödül (reward) |
| !["T(s, a, s^\prime) = P(s^\prime_t=s^\prime \lvert s_t=a, a_t=a)"](https://render.githubusercontent.com/render/math?math=T%28s%2c%20a%2c%20s%5e%5cprime%29%20%3d%20P%28s%5e%5cprime_t%3ds%5e%5cprime%20%5clvert%20s_t%3da%2c%20a_t%3da%29) | !["s_t"](https://render.githubusercontent.com/render/math?math=s_t) durumunda !["a_t"](https://render.githubusercontent.com/render/math?math=a_t) kararı ile !["s_{t+1}"](https://render.githubusercontent.com/render/math?math=s_%7bt%2b1%7d) durumuna geçme olasılığı (transition matrix) |
| !["\pi(s, a) = P(a_t=a \lvert s_t=s)"](https://render.githubusercontent.com/render/math?math=%5cpi%28s%2c%20a%29%20%3d%20P%28a_t%3da%20%5clvert%20s_t%3ds%29) | ajanın !["s_t"](https://render.githubusercontent.com/render/math?math=s_t) durumunda !["a_t"](https://render.githubusercontent.com/render/math?math=a_t) kararını verme olasılığı (policy) |

### Q Öğrenmesi

Pekişirmeli öğrenme başlığında öğrendğimiz determinisik sistemler için *Bellman en iyi çözüm eşitliği* aşağıda verilmiştir. İfadelerde basitlik sağlaması açısından !["Q^{\pi^\ast} (s,a)"](https://render.githubusercontent.com/render/math?math=Q%5e%7b%5cpi%5e%5cast%7d%20%28s%2ca%29) ifadesi yerine !["Q (s,a)"](https://render.githubusercontent.com/render/math?math=Q%20%28s%2ca%29) tercih edilmiştir.

!["Q (s,a) = r(s, a, s^\prime) + \gamma \max_{a^\prime} Q (s^\prime, a^\prime) \label{bellmanOptimality} \tag{1}"](https://render.githubusercontent.com/render/math?math=Q%20%28s%2ca%29%20%3d%20r%28s%2c%20a%2c%20s%5e%5cprime%29%20%2b%20%5cgamma%20%5cmax_%7ba%5e%5cprime%7d%20Q%20%28s%5e%5cprime%2c%20a%5e%5cprime%29%20%5clabel%7bbellmanOptimality%7d%20%5ctag%7b1%7d)

Bu eşitliği çözmek için 1988 yılında Richard Sutton tarafından *"Learning to Predict by the Methods of Temporal Differences"* makalesi ile önerilen TD Öğrenme (Temporal Difference Learning) yöntemi kullanılacaktır. Yazarının da makalenin girişinde belirttiği gibi makalede önerilen yöntem, tahmin etmeyi öğrenme problemini çözme üzerinde kurulmuştur. Yöntem kendi yarattığı, kısmen eksik geçmiş tecrübelerini kullanarak, gelecekteki davranışları tahmin etmeyi amaçlamaktadır. 

Şimdi bu yöntemin Q öğrenmesinde nasıl kullanılacağını inceleyelim. Amacımız !["Q (s,a)"](https://render.githubusercontent.com/render/math?math=Q%20%28s%2ca%29) fonksiyonunun değerini hesaplamak. Bu değeri ajan her !["(s_t=s,a_t=a)"](https://render.githubusercontent.com/render/math?math=%28s_t%3ds%2ca_t%3da%29) durum-aksiyon kararını aldığında elde ettiği ortalama !["\hat{Q}(s,a)"](https://render.githubusercontent.com/render/math?math=%5chat%7bQ%7d%28s%2ca%29) olarak düşünebiliriz. Ancak ortalamanın hesaplanabilmesi için farklı !["t=n, n \in N"](https://render.githubusercontent.com/render/math?math=t%3dn%2c%20n%20%5cin%20N) zamanları için !["(s_n=s,a_n=a)"](https://render.githubusercontent.com/render/math?math=%28s_n%3ds%2ca_n%3da%29) olmasını beklememiz gereklidir. TD öğrenme yöntemi bu ortalama değerin süreç içerisinde gelen her yeni veri ile nasıl güncellenmesi gerektiğini göstermiştir. Sözel olarak anlattığımız algoritmanın matematiksel ispatı şu şekildedir. 

Ortalama hesaplamak istediğimiz !["N"](https://render.githubusercontent.com/render/math?math=N) değerimiz olduğunu varsayalım. Bu durumda ortalama kalite değeri

!["\hat{Q}_N (s,a) = \frac{1}{N} \sum_{n=1}^{N} Q_n(s,a)"](https://render.githubusercontent.com/render/math?math=%5chat%7bQ%7d_N%20%28s%2ca%29%20%3d%20%5cfrac%7b1%7d%7bN%7d%20%5csum_%7bn%3d1%7d%5e%7bN%7d%20Q_n%28s%2ca%29)

olacaktır. Denklemde !["n=N-1"](https://render.githubusercontent.com/render/math?math=n%3dN-1) için eklenen ödülü (son gelen ödül) toplam formulünün dışına alırsak

!["\hat{Q}_N (s,a) = \frac{1}{N} \left ( Q_N(s,a) + \sum_{n=1}^{N-1} Q_n(s,a) \right ) \label{incrementalAvg} \tag{2}"](https://render.githubusercontent.com/render/math?math=%5chat%7bQ%7d_N%20%28s%2ca%29%20%3d%20%5cfrac%7b1%7d%7bN%7d%20%5cleft%20%28%20Q_N%28s%2ca%29%20%2b%20%5csum_%7bn%3d1%7d%5e%7bN-1%7d%20Q_n%28s%2ca%29%20%5cright%20%29%20%5clabel%7bincrementalAvg%7d%20%5ctag%7b2%7d)

elde edilir. Denklemde sağ taraftaki toplam formulüne dikkat edilirse bu formulün !["(N-1)Q_{N-1} (s,a)"](https://render.githubusercontent.com/render/math?math=%28N-1%29Q_%7bN-1%7d%20%28s%2ca%29) olduğu görülür. Bu ifade açık bir şekilde Denklem \ref{incrementalAvg} de yerine yazılırsa

!["\hat{Q}_N (s,a) = \frac{1}{N} \left ( Q_N(s,a) + NQ_{N-1} (s,a) - Q_{N-1} (s,a) \right )"](https://render.githubusercontent.com/render/math?math=%5chat%7bQ%7d_N%20%28s%2ca%29%20%3d%20%5cfrac%7b1%7d%7bN%7d%20%5cleft%20%28%20Q_N%28s%2ca%29%20%2b%20NQ_%7bN-1%7d%20%28s%2ca%29%20-%20Q_%7bN-1%7d%20%28s%2ca%29%20%5cright%20%29)

bulunur. Gerekli sadeleştirmeler yapıldığı takdirde

!["\hat{Q}_N (s,a) = Q_{N-1} (s,a) + \alpha(N) \left ( Q_N(s,a) - Q_{N-1} (s,a) \right ) \label{incrementalAvgFinal} \tag{3}"](https://render.githubusercontent.com/render/math?math=%5chat%7bQ%7d_N%20%28s%2ca%29%20%3d%20Q_%7bN-1%7d%20%28s%2ca%29%20%2b%20%5calpha%28N%29%20%5cleft%20%28%20Q_N%28s%2ca%29%20-%20Q_%7bN-1%7d%20%28s%2ca%29%20%5cright%20%29%20%5clabel%7bincrementalAvgFinal%7d%20%5ctag%7b3%7d)

elde edilir. Burada !["\alpha(N)=\frac{1}{N}"](https://render.githubusercontent.com/render/math?math=%5calpha%28N%29%3d%5cfrac%7b1%7d%7bN%7d) seçilmesi durumunda doğrudan ortalama elde edilirken, !["\alpha(N)"](https://render.githubusercontent.com/render/math?math=%5calpha%28N%29) öğrenme katsayısı olarak düşünülerek, süreç boyunca farklı değerlere ayarlanarak ortalamanın hesaplandığı pencere aralığı da değiştirilebiir olacaktır. Denklem \ref{incrementalAvgFinal} ile verilen iteratif hesaplama yöntemi TD(0) olarak da bilinen TD(!["\lambda"](https://render.githubusercontent.com/render/math?math=%5clambda)) öğrenme algoritmalarının !["\lambda=0"](https://render.githubusercontent.com/render/math?math=%5clambda%3d0) için hesaplanan özel bir değeridir. Q öğrenme algoritması için TD(0) algoritması yeterli olduğundan *"Learning to Predict by the Methods of Temporal Differences"* makalesinde önerilen ve daha geniş bir kapsama alanı bulunan algoritmaya bu yazıda değinilmeyecektir.

Q öğrenmesi işleminin tamamlanması için Denklem \ref{incrementalAvgFinal} ile verilen eşitlikte !["Q_N(s,a)"](https://render.githubusercontent.com/render/math?math=Q_N%28s%2ca%29) yerine Denklem \ref{bellmanOptimality} ile verilen eşitlik yazılırsa Q öğrenmesi işlemi tamamlanmış olur.

!["\hat{Q}_N (s,a) = Q_{N-1} (s,a) + \alpha(N) \left ( r(s, a, s^\prime) + \gamma \max_{a^\prime} Q (s^\prime, a^\prime) - Q_{N-1} (s,a) \right ) \label{TDLearning} \tag{4}"](https://render.githubusercontent.com/render/math?math=%5chat%7bQ%7d_N%20%28s%2ca%29%20%3d%20Q_%7bN-1%7d%20%28s%2ca%29%20%2b%20%5calpha%28N%29%20%5cleft%20%28%20r%28s%2c%20a%2c%20s%5e%5cprime%29%20%2b%20%5cgamma%20%5cmax_%7ba%5e%5cprime%7d%20Q%20%28s%5e%5cprime%2c%20a%5e%5cprime%29%20-%20Q_%7bN-1%7d%20%28s%2ca%29%20%5cright%20%29%20%5clabel%7bTDLearning%7d%20%5ctag%7b4%7d)

Bu işlem ajan çevreyi öğrenmeye devam ettiği sürece tekrarlanarak Q tablosunun öğrenilmesi sağlanır.

### Q Öğrenmesi C Kodu

Yukarıda bahsedilen Q öğrenme algortimasının güncelleme adımı aşağıdaki C kodu ile gerçeklenmiştir. Denklemden de görüleceği üzere güncelleme adımında !["(s,a,s')"](https://render.githubusercontent.com/render/math?math=%28s%2ca%2cs%27%29) değerlerine ihtiyaç vardır. Bunlara ek olarak unutma katsayısı ve bu ödül de fonksiyona girdi olarak verilmiştir.

```c
// compute the update for the Q table
void q_table_update(struct q_table_t *Q, uint32_t s, uint32_t a, uint32_t sp, float alpha, float reward)
{
    uint32_t a = 0;

    // find the maximum q in the next state
    float max_q = Q->qtable[sp][0];
    for(a = 1; a < Q->num_actions; a++)
    {
        max_q = max(max_q, Q->qtable[sp][a]);
    }

    // update the table
    Q->qtable[s][a] += alpha * (reward + Q->gamma * max_q - Q->qtable[s][a]);
}
```

Q öğrenmede durumlar arasında geçiş için kullanılan !["\pi^\ast (s) = \arg\max_{a} Q^\pi (s,a)"](https://render.githubusercontent.com/render/math?math=%5cpi%5e%5cast%20%28s%29%20%3d%20%5carg%5cmax_%7ba%7d%20Q%5e%5cpi%20%28s%2ca%29) fonksiyonu `q_table_get_action` yöntemi ile gerçeklenmiştir. Verilen bir durum ve keşfetme katsayısı ile fonksiyon yeni bir yol mu deneyeceğine yoksa bildiği en iyi yoldan mı gideceğine karar verdikten sonra duruma uygun bir aksiyonu seçmektedir.

```c
// get the action proposed by the q learning algorithm
uint32_t q_table_get_action(struct q_table_t *Q, uint32_t s, float exploration)
{
    // pick a random action
    uint32_t max_a = random_int(0, Q->num_actions - 1);

    // if in exploatation mode, find the best move
    if(random_float(0,1) > exploration)
    {
        // find the maximum q in the next state
        uint32_t a = 0;
        for (a = 0; a < Q->num_actions; a++)
        {
            if (Q->qtable[s][a] > Q->qtable[s][max_a])
            {
                max_a = a;
            }
        }
    }
    
    // return the action
    return max_a;
}
```

Burada kullanılan `exploration` katsayısı ajanın öğrenme sırasında olabildiğince fazla durumu keşfetmesini sağlamak için yapılan bir ilavedir. `exploration` katsayısının bire yaklaştığı durumlarda ajan, en iyi yolu izlemek yerine rastgele bir yolu seçecektir. Bu katsayı özellikle öğrenme işleminin başlarında tablonun tüm durum-aksiyon çiftlerinin güncellenebilmesi için gerekli bir değişkendir. Öğrenme safhası belirli bir noktaya gelip, !["Q(s,a)"](https://render.githubusercontent.com/render/math?math=Q%28s%2ca%29) değerleri ideal duruma yaklaştığında bu değer azaltılarak ajanın en ideal durum-aksiyon çiftlerini öğrenmesi sağlanır.

### Pekiştirmeli Öğrenme ile Oyun Programlama

Şimdi basit bir oyunu pekiştirmeli öğrenme kullanarak nasıl öğrenebileceğimize bir bakalım. Aşağıdaki grafikte !["5\times 7"](https://render.githubusercontent.com/render/math?math=5%5ctimes%207) boyutlu bir ızagara dünyası verilmiştir.

![Pekiştirmeli Öğrenme ile Oyun Programlama#half][simple_game_diagram]

Ajanımız amacı bu dünyadan sol,yukarı,sağ veya aşağı yönlü hareketler yaparak çıkmaya çalışmak olacak. Izgara dünyasından çıkış için kullanılabilecek toplam beş kapı bulunmakta. Ajan grafikte siyah ile gösterilen kapılardan çıkması durumunda !["P=r(s \in \\{ 9,11,23,25\\})=-1.0"](https://render.githubusercontent.com/render/math?math=P%3dr%28s%20%5cin%20%5c%5c%7b%209%2c11%2c23%2c25%5c%5c%7d%29%3d-1.0) ödülünü alırken, yeşil ile gösterilen çıkış kapısından çıkması durumunda !["R=r(s \in \\{ 17 \\}=1.0"](https://render.githubusercontent.com/render/math?math=R%3dr%28s%20%5cin%20%5c%5c%7b%2017%20%5c%5c%7d%3d1.0) ödülünü alacaktır. Ajanın ızgara dünyasından çıkışını hızlandırmak için ajanımız yaptığı her hareket sonucunda eğer siyah veya yeşil karelere gelemedi ise !["T=r(s \not\in \\{ 9,11,17,23,25\\})=-0.1"](https://render.githubusercontent.com/render/math?math=T%3dr%28s%20%5cnot%5cin%20%5c%5c%7b%209%2c11%2c17%2c23%2c25%5c%5c%7d%29%3d-0.1) ceza alacaktır.

Bu şartlar altında çevreyi keşfederek her durum için alması gereken kararı öğrenen pekiştirmeli öğrenme algoritması C dilinde yazılmıştır. Yazılan kodun tamamı projenin [GitHub](https://github.com/cescript/reinforcement_learning_q_learning) sayfası üzerinden incelenebilir. Burada pekiştirmeli öğrenmenin en önemli kısmını içeren ve öğrenmenin yapıldığı kod parçası verilmiştir.

```c
// play single game and return the game result
int play_game(struct q_table_t *Q, struct game_t *game, float exploration)
{
    uint32_t current_state = get_state(game);

    // play a single game untill the end
    while (!game->isEndS[current_state])
    {
        uint32_t action = q_table_get_action(Q, current_state, exploration);

        // update the target position
        update_target_position(game, action);

        // get the next state after the selected action
        uint32_t next_state = get_state(game);

        // do learning
        q_table_update(Q, current_state, action, next_state, 0.01, game->reward[next_state]);

        // goto next state
        current_state = next_state;
    }

    return game->reward[current_state] == R ? 1:0;
}
```
Verilen `play_game` fonksiyonu farklı `exploration` katsayıları ile 100 bin kez çağrılarak girdi olarak verilen `Q` tablsonun öğrenilmesi sağlanmıştır. Her bir çağrıda; ilk olarak ajanın bulunduğu durum hesaplanmış ve bu durum için en uygun hareket yukarıda gerçeklemesi verilen `q_table_get_action` yöntemi ile bulunmuştur. Ardından ajanın bu hareket ile gideceği yeni durum `next_state` hesaplanmış ve Denklem \ref{incrementalAvgFinal} ile verilen matematiksel ifadenin gerçeklendiği `q_table_update` yöntemi ile pekiştirmeli öğrenme gerçeklenmiştir. 

Bu öğrenme işlemi sonrasında ajan her durum için hangi kararın ne kadar kazançlı olduğunu gösteren `Q` tablosuna sahip olacaktır. Q öğrenmenin de temelinde olduğu gibi !["\pi^\ast (s) = \arg\max_{a} Q^\pi (s,a)"](https://render.githubusercontent.com/render/math?math=%5cpi%5e%5cast%20%28s%29%20%3d%20%5carg%5cmax_%7ba%7d%20Q%5e%5cpi%20%28s%2ca%29) kuralına göre hareket eden bir ajan için hangi durumda hangi yöne gitmesini gösteren tablo aşağıdaki grafikte verilmiştir.

![Pekiştirmeli Öğrenme ile Oyun Programlama#half][simple_game_diagram_learned]

Grafikten de görüldüğü üzere ajan her durum için alması gereken doğru kararı başarılı bir şekilde öğrenmiştir. Serinin bir sonraki yazısında daha karmaşık bir problemin pekiştirmeli öğrenme problemine nasıl dönüştüreleceği incelenecektir.

**Referanslar**
* Mnih, Volodymyr, et al. "Playing atari with deep reinforcement learning." arXiv preprint arXiv:1312.5602 (2013).
* Sutton, Richard S. "Learning to predict by the methods of temporal differences." Machine learning 3.1 (1988): 9-44.
* Watkins, Christopher JCH, and Peter Dayan. "Q-learning." Machine learning 8.3-4 (1992): 279-292.
* Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.


[RESOURCES]: # (List of the resources used by the blog post)
[simple_game_diagram]: /assets/post_resources/reinforcement_learning_q_learning/simple_game_diagram.svg
[simple_game_diagram_learned]: /assets/post_resources/reinforcement_learning_q_learning/simple_game_diagram_learned.svg
