from manim import *

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Historical Evolution: The Polygon Method", [
            "Ancient math masters estimated pi long ago.",
            "They used polygons inside and outside circles.",
            "More sides meant a better, tighter fit.",
            "This method exhausted the area difference.",
            "Geometry paved the way for our understanding."
        ])
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        # Load asset per B018
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        circle = Circle(radius=1.2, color="#FFFFFF")
        label_c = Text("C", font_size=24, color="#FFFFFF")
        
        self.place_at_grid(circle, 'C5', scale_factor=0.6)
        self.place_at_grid(compass, 'B5', scale_factor=0.5)
        self.place_at_grid(label_c, 'B3', scale_factor=0.6)
        self.play(FadeIn(circle), FadeIn(compass), Write(label_c))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF00FF")
        polygon = RegularPolygon(n=4, radius=1.2, color="#FF00FF")
        self.place_at_grid(polygon, 'C5', scale_factor=1.0)
        self.play(Create(polygon))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF00FF")
        poly_6 = RegularPolygon(n=6, radius=1.2, color="#FF00FF")
        poly_12 = RegularPolygon(n=12, radius=1.2, color="#FF00FF")
        self.place_at_grid(poly_6, 'C5', scale_factor=1.0)
        self.place_at_grid(poly_12, 'C5', scale_factor=1.0)
        self.play(Transform(polygon, poly_6))
        self.play(Transform(polygon, poly_12))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#00FFFF")
        label_ca = Text("Circle Area", font_size=18, color="#00FFFF")
        label_pa = Text("Poly Area", font_size=18, color="#00FFFF")
        self.place_at_grid(label_ca, 'D2', scale_factor=0.8)
        self.place_at_grid(label_pa, 'D5', scale_factor=0.8)
        self.play(Write(label_ca), Write(label_pa))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFFF00")
        scroll = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/scroll.svg")
        final_poly = RegularPolygon(n=50, radius=1.2, color="#FFFF00")
        self.place_at_grid(final_poly, 'C5', scale_factor=1.0)
        self.place_at_grid(scroll, 'E5', scale_factor=0.6)
        self.play(Transform(polygon, final_poly), FadeIn(scroll), run_time=2)
        self.wait(1)
