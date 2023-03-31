
'''Python içine gerekli kütüphaneler import edilir.Diğer kütüphaneler ilgili işlem yapılmadan önce aşağıda import edilecektir. '''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import warnings
warnings.filterwarnings('ignore')

'''Kullanacağımız veri seti Iris veri setidir.Python'ın pandas kütüphanesinin read_excel methodu ile veri setini import ediyoruz.'''

data = pd.read_excel("Iris.xls")

print(data) 

'''Veri setinde yer alan kolonlarda bağımlı değşiken kolonuna ait belirli özellikler bulunmaktadır.Bu özelliklerden birkaçını kullanarak makine öğrenmesinde gözetimsiz öğrenme 
yöntemlerinden biri olan K-means algoritması ile çalışacağız.Özelliklerimizden bir matris oluşturup v değişkenine atadık.'''

v=data.iloc[:,1:4].values

'''Makine öğrenmesi gözetimsiz öğrenme yöntemlerinden biri de Hiyerarşik bölütlemedir.Hiyerarşik bölütleme,bir grup veriyi bir hiyerarşi yapısında küçük alt gruplara bölme işlemidir.
Hiyerarşik bölütleme, aglomeratif ve bölücü olmak üzere iki farklı yaklaşım kullanır. Aglomeratif yaklaşım, başlangıçta her bir verinin ayrı bir küme olarak ele alındığı ve
 ardından benzer kümeleme özelliklerine sahip verilerin birleştirildiği bir yöntemdir. Bölücü yaklaşım ise başlangıçta tüm verilerin tek bir küme olarak ele alındığı ve ardından 
 verilerin benzer olmayan özelliklere sahip alt gruplara ayrıldığı bir yöntemdir.Genellikle kümeleme problemlerinde ise aglomeratif yaklaşım tercih edilmekte olup yapılan çalışmada da 
 bu yöntem kullanılacaktır.Öncelikle kullanılacak algoritma Scikit-Learn kütüphanesinden import edilir.'''
 

from sklearn.cluster import AgglomerativeClustering


'''Kullanılacak aglomeratif algoritmasında küme sayısının optimal belirlenmesi için kullanılan araç Dendogram Grafiğidir.Dendogram grafiği,verilerin hiyerarşik olarak görselleştirilmesini
sağlar.Bu yöntem verilerin kolay yorumlanmasını sağlarken küme sayısını da optimal olarak belirlememize yardımcı olur.Dendogram grafiğinin verimli olarak kullanılması için belli 
parametrelerin verilmesi gerekir.Dendogram grafiği çizildiğinde görülecektir ki veriler arasında belirli bağlantılar ile kümeler oluşmaktadır.Farklı farklı bağlantı türleri olup
bu çalışmada ward bağlantı yöntemi kullanılacaktır.Ward bağlantısı, hiyerarşik kümeleme yöntemlerinden biridir ve kümeleme işlemindeki iki kümenin birleştirilmesi için kullanılan 
bir kriterdir.'''

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(v,method='ward'))
plt.show()

'''Dendogram grafiği üzerinde kaç küme oluşturacağımız ile ilgili yorumlamada bulunabiliriz.Oluşturulan dendogram grafiğine göre optimal olarak k değerini 3 olarak belirleriz.Mesafe ölçümü
için öklid metodu kullanırken bağlantı parametresi için ise yine ward metodundan faydalanacağız.'''

ac = AgglomerativeClustering(n_clusters=3, affinity='euclidean',linkage='ward')
predict = ac.fit_predict(v)

'''Makinenin tahminlerinin ayrı ayrı grafikte gösterimi için aşağıda saçılım grafiğine yer verilmiştir.'''
plt.scatter(v[predict==0,0],v[predict==0,1],s=60,color='pink')
plt.scatter(v[predict==1,0],v[predict==1,1],s=60,color='blue')
plt.scatter(v[predict==2,0],v[predict==2,1],s=60,color='green')
plt.title('Hierarchical Clustering')
plt.show()
