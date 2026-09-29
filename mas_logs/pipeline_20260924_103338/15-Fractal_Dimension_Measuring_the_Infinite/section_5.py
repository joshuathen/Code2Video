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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Nature utilizes fractals for efficient surface area.",
            "Bronchial trees maximize gas exchange through complexity.",
            "Dimension helps diagnose health and monitor systems."
        ]
        self.setup_layout("Real-World Application: Nature's Geometry", lecture_lines)
        
        # Colors
        color1 = "#4CAF50" # Nature green
        color2 = "#7FFF00" # Bronchial neon green (per storyboard)
        color3 = "#FF9800" # Health orange

        # Assets
        lung_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/lung.svg"
        lung_obj = SVGMobject(lung_path).set_color(color2)
        
        # 1. Nature fractal representation
        nature_obj = VGroup(
            *[Triangle().scale(0.3).set_color(color1) for _ in range(5)]
        ).arrange_in_grid(rows=1, cols=5)
        
        # 3. Fractal dimension label
        dim_label = MathTex(r"D \approx 2.7", color=color3).scale(1.2)
        dim_label_text = Text("Dimension", font_size=18, color=color3)
        dim_group = VGroup(dim_label, dim_label_text).arrange(DOWN)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(color1)
        self.place_in_area(nature_obj, 'A4', 'B6', scale_factor=0.6)
        self.play(FadeIn(nature_obj))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(color2)
        self.play(FadeOut(nature_obj))
        self.place_at_grid(lung_obj, 'C5', scale_factor=0.8)
        self.play(Create(lung_obj))
        self.play(lung_obj.animate.scale(1.2)) # Expand effect
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(color3)
        self.place_at_grid(dim_group, 'E5', scale_factor=0.9)
        self.play(Write(dim_group))
        self.wait(1)
