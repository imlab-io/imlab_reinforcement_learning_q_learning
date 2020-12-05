---
layout: post
title: Q Öğrenme
slug: q-learning
author: Bahri ABACI
categories:
- Makine Öğrenmesi
- Nümerik Yöntemler
- Veri Analizi
references: ""
thumbnail: /assets/post_resources/reinforcement_learning_q_learning/thumbnailq.png
---
İlk olarak Christopher JCH Watkins and Peter Dayan tarafından 1992 yılında literatüre kazandırılan Q Öğrenme (Q Learning) yöntemi, 2013 yılında [DeepMind](https://deepmind.com) yapay zeka şirketinin kurucuları tarafından yayınlanan *"Playing atari with deep reinforcement learning"* makalesi ile oldukça popüler hale gelen bir pekiştirmeli öğrenme yöntemidir. Algoritmanın matematiksel temellerinin oluşturulduğu [Pekiştirmeli Öğrenme]({% post_url 2020-01-05-pekistirmeli-ogrenme %}) yazımızda da bahsedildiği gibi pekiştirmeli öğrenme, [K-Means]({% post_url 2015-08-28-k-means-kumeleme-algoritmasi %}), [Temel Bileşen Analizi]({% post_url 2019-09-01-temel-bilesen-analizi %}), [Karar Ağaçları]({% post_url 2015-11-01-karar-agaclari %}), [Lojistik Regresyon Analizi]({% post_url 2015-07-23-lojistik-regresyon-analizi %}) ve Destek Vektör Makineleri gibi yöntemlerin hepsinden farklı olarak herhangi bir veriye ihtiyaç duymadan öğrenme yapabilmektedir. Yöntem verilen kurallar çerçevesinde durum uzayını (state-space) keşfederek her durum için elde edeceği ödülü en büyüklemesini sağlayan hareketi (action) öğrenmeye çalışmaktadır.

<!--more-->

Bu yazımızda [Pekiştirmeli Öğrenme]({% post_url 2020-01-05-pekistirmeli-ogrenme %}) yazımızda türetilen Q öğrenme yönteminin matematiksel ifadesi kullanılarak Q öğrenme algoritması incelenecek ve kodlanacaktır. Eğer pekiştirmeli öğrenme mantığı veya burada kullanılacak olan Bellman denklemi, kalite fonksiyonu gibi denklemler hakkında eksiğiniz olduğunu düşünüyorsanız öncelikle [bir önceki yazıyı]({% post_url 2020-01-05-pekistirmeli-ogrenme%}) okumanızı tavsiye ederim.

Öğrenme işleminin matematiksel olarak nasıl yapıldığına geçmeden önce, kullanacağımız matematiksel sembolleri ve anlamlarını bir tablo şeklinde yazalım.

| Sembol | Açıklama |
:-------------------------:|:-------------------------
| <img src="assets/post_resources/math//1f1c28e0a1b1708c6889fb006c886784.svg?invert_in_darkmode" align=middle width=12.67127234999999pt height=14.15524440000002pt/> | <img src="assets/post_resources/math//4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode" align=middle width=5.936097749999991pt height=20.221802699999984pt/> anındaki durum (state) |
| <img src="assets/post_resources/math//208ea3d3be1f9f16de483eb512e60c84.svg?invert_in_darkmode" align=middle width=46.89677849999999pt height=22.465723500000017pt/> | <img src="assets/post_resources/math//53d147e7f3fe6e47ee05b88b166bd3f6.svg?invert_in_darkmode" align=middle width=12.32879834999999pt height=22.465723500000017pt/> hareketler uzayından alınan bir hareket (action) |
| <img src="assets/post_resources/math//cca77a5dd11dbadda29249627731c3f8.svg?invert_in_darkmode" align=middle width=64.72599044999998pt height=24.7161288pt/> | <img src="assets/post_resources/math//4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode" align=middle width=5.936097749999991pt height=20.221802699999984pt/> anında <img src="assets/post_resources/math//44bc9d542a92714cac84e01cbbb7fd61.svg?invert_in_darkmode" align=middle width=8.68915409999999pt height=14.15524440000002pt/> hareketi ile geçilen yeni durum (new state) |
| <img src="assets/post_resources/math//3046740be7e991536cad21a5cc767431.svg?invert_in_darkmode" align=middle width=111.85497014999997pt height=24.7161288pt/> | <img src="assets/post_resources/math//1f1c28e0a1b1708c6889fb006c886784.svg?invert_in_darkmode" align=middle width=12.67127234999999pt height=14.15524440000002pt/> durumunda <img src="assets/post_resources/math//9789555e5d8fa5de21171cc40c86d2cd.svg?invert_in_darkmode" align=middle width=13.65494624999999pt height=14.15524440000002pt/> kararı ile <img src="assets/post_resources/math//395ed339b8b48a682566a5d43c12344f.svg?invert_in_darkmode" align=middle width=12.67127234999999pt height=24.7161288pt/> durumuna geçilerek elde edilen ödül (reward) |
| <img src="assets/post_resources/math//11e6bb9296d9e258b967741389d2a655.svg?invert_in_darkmode" align=middle width=264.32217239999994pt height=24.7161288pt/> | <img src="assets/post_resources/math//1f1c28e0a1b1708c6889fb006c886784.svg?invert_in_darkmode" align=middle width=12.67127234999999pt height=14.15524440000002pt/> durumunda <img src="assets/post_resources/math//9789555e5d8fa5de21171cc40c86d2cd.svg?invert_in_darkmode" align=middle width=13.65494624999999pt height=14.15524440000002pt/> kararı ile <img src="assets/post_resources/math//b02dd36b8a10566f2a0ad9cbb2e74858.svg?invert_in_darkmode" align=middle width=29.31519194999999pt height=14.15524440000002pt/> durumuna geçme olasılığı (transition matrix) |
| <img src="assets/post_resources/math//eb3ddcc22d5eb7d908d2582d64b750a3.svg?invert_in_darkmode" align=middle width=186.75200114999998pt height=24.65753399999998pt/> | ajanın <img src="assets/post_resources/math//1f1c28e0a1b1708c6889fb006c886784.svg?invert_in_darkmode" align=middle width=12.67127234999999pt height=14.15524440000002pt/> durumunda <img src="assets/post_resources/math//9789555e5d8fa5de21171cc40c86d2cd.svg?invert_in_darkmode" align=middle width=13.65494624999999pt height=14.15524440000002pt/> kararını verme olasılığı (policy) |

### Q Öğrenmesi

Pekişirmeli öğrenme başlığında öğrendğimiz determinisik sistemler için *Bellman en iyi çözüm eşitliği* aşağıda verilmiştir. İfadelerde basitlik sağlaması açısından <img src="assets/post_resources/math//f537098c46a2b2d40ec0893cc53e913e.svg?invert_in_darkmode" align=middle width=65.27545034999999pt height=29.50059090000001pt/> ifadesi yerine <img src="assets/post_resources/math//6177767dc0d2ca910cc18958a6855180.svg?invert_in_darkmode" align=middle width=49.48137479999998pt height=24.65753399999998pt/> tercih edilmiştir.

<p align="center"><img src="assets/post_resources/math//659005a6db8fc7327d5f121ea27b7010.svg?invert_in_darkmode" align=middle width=259.6743567pt height=23.77356135pt/></p>

Bu eşitliği çözmek için 1988 yılında Richard Sutton tarafından *"Learning to Predict by the Methods of Temporal Differences"* makalesi ile önerilen TD Öğrenme (Temporal Difference Learning) yöntemi kullanılacaktır. Yazarının da makalenin girişinde belirttiği gibi makalede önerilen yöntem, tahmin etmeyi öğrenme problemini çözme üzerinde kurulmuştur. Yöntem kendi yarattığı, kısmen eksik geçmiş tecrübelerini kullanarak, gelecekteki davranışları tahmin etmeyi amaçlamaktadır. 

Şimdi bu yöntemin Q öğrenmesinde nasıl kullanılacağını inceleyelim. Amacımız <img src="assets/post_resources/math//6177767dc0d2ca910cc18958a6855180.svg?invert_in_darkmode" align=middle width=49.48137479999998pt height=24.65753399999998pt/> fonksiyonunun değerini hesaplamak. Bu değeri ajan her <img src="assets/post_resources/math//ab7c5ab6404b062d656da894dac08c35.svg?invert_in_darkmode" align=middle width=108.29122094999998pt height=24.65753399999998pt/> durum-aksiyon kararını aldığında elde ettiği ortalama <img src="assets/post_resources/math//cc1827d427b44ce3d6c9f7dc9c020c3e.svg?invert_in_darkmode" align=middle width=49.481371499999995pt height=31.141535699999984pt/> olarak düşünebiliriz. Ancak ortalamanın hesaplanabilmesi için farklı <img src="assets/post_resources/math//b70aaff26a9a3568eb66ab132a79cedd.svg?invert_in_darkmode" align=middle width=89.9844924pt height=22.465723500000017pt/> zamanları için <img src="assets/post_resources/math//5d20dc98fceba8a0f5cf86edc2772cae.svg?invert_in_darkmode" align=middle width=114.61172084999998pt height=24.65753399999998pt/> olmasını beklememiz gereklidir. TD öğrenme yöntemi bu ortalama değerin süreç içerisinde gelen her yeni veri ile nasıl güncellenmesi gerektiğini göstermiştir. Sözel olarak anlattığımız algoritmanın matematiksel ispatı şu şekildedir. 

Ortalama hesaplamak istediğimiz <img src="assets/post_resources/math//f9c4988898e7f532b9f826a75014ed3c.svg?invert_in_darkmode" align=middle width=14.99998994999999pt height=22.465723500000017pt/> değerimiz olduğunu varsayalım. Bu durumda ortalama kalite değeri

<p align="center"><img src="assets/post_resources/math//abea97136bfcdbf0f70eb45df09e9ebd.svg?invert_in_darkmode" align=middle width=191.49076155pt height=47.60747145pt/></p>

olacaktır. Denklemde <img src="assets/post_resources/math//d78ff214efe92742cbec4d6c8d0d50b7.svg?invert_in_darkmode" align=middle width=75.09487589999999pt height=22.465723500000017pt/> için eklenen ödülü (son gelen ödül) toplam formulünün dışına alırsak

<p align="center"><img src="assets/post_resources/math//976b77fd2ea85801fc75f3157e833559.svg?invert_in_darkmode" align=middle width=303.26160645pt height=49.315569599999996pt/></p>

elde edilir. Denklemde sağ taraftaki toplam formulüne dikkat edilirse bu formulün <img src="assets/post_resources/math//921d7f07a022a6beb2e345975e872dbc.svg?invert_in_darkmode" align=middle width=134.87177054999998pt height=24.65753399999998pt/> olduğu görülür. Bu ifade açık bir şekilde Denklem \ref{incrementalAvg} de yerine yazılırsa

<p align="center"><img src="assets/post_resources/math//b779450b837bbcab28624920ce31e033.svg?invert_in_darkmode" align=middle width=393.02100255pt height=32.990165999999995pt/></p>

bulunur. Gerekli sadeleştirmeler yapıldığı takdirde

<p align="center"><img src="assets/post_resources/math//abe9d74ad1501d5f0a1533266032d1ed.svg?invert_in_darkmode" align=middle width=397.43778855pt height=19.68035685pt/></p>

elde edilir. Burada <img src="assets/post_resources/math//4c5d9cd0ee15d594afa1e020f9c32871.svg?invert_in_darkmode" align=middle width=73.89828929999999pt height=27.77565449999998pt/> seçilmesi durumunda doğrudan ortalama elde edilirken, <img src="assets/post_resources/math//a549697596e37c86c5c48a0c8ac20b21.svg?invert_in_darkmode" align=middle width=38.36189939999999pt height=24.65753399999998pt/> öğrenme katsayısı olarak düşünülerek, süreç boyunca farklı değerlere ayarlanarak ortalamanın hesaplandığı pencere aralığı da değiştirilebiir olacaktır. Denklem \ref{incrementalAvgFinal} ile verilen iteratif hesaplama yöntemi TD(0) olarak da bilinen TD(<img src="assets/post_resources/math//fd8be73b54f5436a5cd2e73ba9b6bfa9.svg?invert_in_darkmode" align=middle width=9.58908224999999pt height=22.831056599999986pt/>) öğrenme algoritmalarının <img src="assets/post_resources/math//60b6e7fd8b1dea748d9229c2822315dd.svg?invert_in_darkmode" align=middle width=39.72592304999999pt height=22.831056599999986pt/> için hesaplanan özel bir değeridir. Q öğrenme algoritması için TD(0) algoritması yeterli olduğundan *"Learning to Predict by the Methods of Temporal Differences"* makalesinde önerilen ve daha geniş bir kapsama alanı bulunan algoritmaya bu yazıda değinilmeyecektir.

Q öğrenmesi işleminin tamamlanması için Denklem \ref{incrementalAvgFinal} ile verilen eşitlikte <img src="assets/post_resources/math//985cc75b9a166ed283988fd83fc95a46.svg?invert_in_darkmode" align=middle width=61.949428199999986pt height=24.65753399999998pt/> yerine Denklem \ref{bellmanOptimality} ile verilen eşitlik yazılırsa Q öğrenmesi işlemi tamamlanmış olur.

<p align="center"><img src="assets/post_resources/math//92b5d79ad6bfa546ebddb2b27b0b3cef.svg?invert_in_darkmode" align=middle width=530.6130291000001pt height=29.58934275pt/></p>

Bu işlem ajan çevreyi öğrenmeye devam ettiği sürece tekrarlanarak Q tablosunun öğrenilmesi sağlanır.

### Q Öğrenmesi C Kodu

Yukarıda bahsedilen Q öğrenme algortimasının güncelleme adımı aşağıdaki C kodu ile gerçeklenmiştir. Denklemden de görüleceği üzere güncelleme adımında <img src="assets/post_resources/math//95c0c4b291ddd9c1c39be5b9516adaa6.svg?invert_in_darkmode" align=middle width=56.109187199999994pt height=24.7161288pt/> değerlerine ihtiyaç vardır. Bunlara ek olarak unutma katsayısı ve bu ödül de fonksiyona girdi olarak verilmiştir.

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

Q öğrenmede durumlar arasında geçiş için kullanılan <img src="assets/post_resources/math//12760ccb4534e093c03bc8d6ae07243a.svg?invert_in_darkmode" align=middle width=185.45934989999998pt height=24.65753399999998pt/> fonksiyonu `q_table_get_action` yöntemi ile gerçeklenmiştir. Verilen bir durum ve keşfetme katsayısı ile fonksiyon yeni bir yol mu deneyeceğine yoksa bildiği en iyi yoldan mı gideceğine karar verdikten sonra duruma uygun bir aksiyonu seçmektedir.

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

Burada kullanılan `exploration` katsayısı ajanın öğrenme sırasında olabildiğince fazla durumu keşfetmesini sağlamak için yapılan bir ilavedir. `exploration` katsayısının bire yaklaştığı durumlarda ajan, en iyi yolu izlemek yerine rastgele bir yolu seçecektir. Bu katsayı özellikle öğrenme işleminin başlarında tablonun tüm durum-aksiyon çiftlerinin güncellenebilmesi için gerekli bir değişkendir. Öğrenme safhası belirli bir noktaya gelip, <img src="assets/post_resources/math//a18656976616796e481e7c608b8a2b40.svg?invert_in_darkmode" align=middle width=49.48137479999998pt height=24.65753399999998pt/> değerleri ideal duruma yaklaştığında bu değer azaltılarak ajanın en ideal durum-aksiyon çiftlerini öğrenmesi sağlanır.

### Pekiştirmeli Öğrenme ile Oyun Programlama

Şimdi basit bir oyunu pekiştirmeli öğrenme kullanarak nasıl öğrenebileceğimize bir bakalım. Aşağıdaki grafikte <img src="assets/post_resources/math//32d26491758aac001c93608b2e41be62.svg?invert_in_darkmode" align=middle width=36.52961069999999pt height=21.18721440000001pt/> boyutlu bir ızagara dünyası verilmiştir.

![Pekiştirmeli Öğrenme ile Oyun Programlama#half][simple_game_diagram]

Ajanımız amacı bu dünyadan sol,yukarı,sağ veya aşağı yönlü hareketler yaparak çıkmaya çalışmak olacak. Izgara dünyasından çıkış için kullanılabilecek toplam beş kapı bulunmakta. Ajan grafikte siyah ile gösterilen kapılardan çıkması durumunda <img src="assets/post_resources/math//167088adb1b11907bf4d75f5a06354ef.svg?invert_in_darkmode" align=middle width=141.55249514999997pt height=47.671232400000015pt/> ödülünü alırken, yeşil ile gösterilen çıkış kapısından çıkması durumunda <img src="assets/post_resources/math//15df2a0315118d45b36e2da3524686fd.svg?invert_in_darkmode" align=middle width=104.89903545pt height=39.45205440000001pt/> ödülünü alacaktır. Ajanın ızgara dünyasından çıkışını hızlandırmak için ajanımız yaptığı her hareket sonucunda eğer siyah veya yeşil karelere gelemedi ise <img src="assets/post_resources/math//547e3fea634e558551210b176d5164bd.svg?invert_in_darkmode" align=middle width=165.29678385pt height=47.671232400000015pt/> ceza alacaktır.

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

Bu öğrenme işlemi sonrasında ajan her durum için hangi kararın ne kadar kazançlı olduğunu gösteren `Q` tablosuna sahip olacaktır. Q öğrenmenin de temelinde olduğu gibi <img src="assets/post_resources/math//12760ccb4534e093c03bc8d6ae07243a.svg?invert_in_darkmode" align=middle width=185.45934989999998pt height=24.65753399999998pt/> kuralına göre hareket eden bir ajan için hangi durumda hangi yöne gitmesini gösteren tablo aşağıdaki grafikte verilmiştir.

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
