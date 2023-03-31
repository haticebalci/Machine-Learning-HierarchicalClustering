# Machine-Learning-HierarchicalClustering
# HierarchicalClustering
Yukarıdaki kod bloğunda, Iris veri seti üzerinde hiyerarşik bölütleme algoritması kullanılarak kümeleme yapılıyor. Kodda kullanılan kütüphaneler arasında Pandas, Numpy, Matplotlib ve Scikit-learn yer alıyor.

Kodun açıklaması şu şekilde:

İlk olarak gerekli kütüphaneler import ediliyor ve Iris veri seti, Pandas kütüphanesi kullanılarak okunuyor.
Ardından, veri setinden belirli sütunlardan bir matris oluşturuluyor ve v değişkenine atanıyor.
Daha sonra, kullanılacak hiyerarşik bölütleme algoritması olan AgglomerativeClustering Scikit-learn kütüphanesi kullanılarak import ediliyor.
Küme sayısının optimal olarak belirlenmesi için dendogram grafiği oluşturuluyor. Bu grafiğin oluşturulması için, Scipy kütüphanesindeki cluster.hierarchy modülü kullanılarak dendrogram fonksiyonu çağrılıyor.
Oluşturulan dendogram grafiği üzerindeki yorumlamalar sonucunda, k değeri olarak 3 belirleniyor.
Son olarak, algoritma n_clusters=3, affinity='euclidean',linkage='ward' parametreleriyle çalıştırılıyor ve sonuçlar matplotlib kütüphanesi kullanılarak scatter plot grafiğiyle görselleştiriliyor.
