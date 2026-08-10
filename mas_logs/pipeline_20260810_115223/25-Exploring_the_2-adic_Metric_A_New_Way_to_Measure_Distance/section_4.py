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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Visualizing Divergence vs. Convergence", [
            "Compare standard divergence versus 2-adic convergence.",
            "Ultrametric inequality keeps triangles isosceles here.",
            "1 plus 2 plus 4 equals negative 1."
        ])
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        
        # Load assets
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg", color=BLUE)
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg", color=BLUE)
        
        # Euclidean side (left)
        eucl_label = Text("Euclidean", font_size=20)
        eucl_group = VGroup(eucl_label, ruler).arrange(DOWN)
        # Using suggested fix for issue 34
        self.place_in_area(eucl_group, "A2", "B3", scale_factor=0.3)
        self.add(eucl_group)

        # 2-adic side (right)
        padic_label = Text("2-adic", font_size=20)
        padic_group = VGroup(padic_label, compass).arrange(DOWN)
        # Using suggested fix for issue 35
        self.place_in_area(padic_group, "A5", "B6", scale_factor=0.3)
        self.add(padic_group)
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(GREEN)
        
        tri = Triangle(color=WHITE).scale(0.5)
        tri_label = Text("Isosceles Triangle", font_size=18)
        tri_group = VGroup(tri, tri_label).arrange(DOWN)
        self.place_in_area(tri_group, "D2", "E3", scale_factor=0.3)
        self.add(tri_group)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        
        sum_text = MathTex(r"1+2+4+\dots = -1", font_size=28, color=YELLOW)
        # Using suggested fix for issue 36
        self.place_at_grid(sum_text, "D5", scale_factor=0.6)
        self.add(sum_text)
        self.wait()
