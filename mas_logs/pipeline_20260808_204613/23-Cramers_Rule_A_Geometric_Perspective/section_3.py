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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Replace column a1 with target vector b.",
            "New parallelogram formed by b and a2.",
            "Ratio of areas gives scaling factor x.",
            "Cramer's Rule equals area ratio calculation.",
            "This reveals x coordinate geometrically."
        ]
        self.setup_layout("Geometric Derivation of Cramer's Rule", lecture_lines)
        
        # Colors
        colors = ["#FF7F50", "#87CEFA", "#90EE90", "#FFD700", "#FF69B4"]
        
        # --- Define Assets ---
        grid_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        calc_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg")
        prot_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/protractor.svg")

        # --- Define Mobjects ---
        # Fixed Grid Plane
        plane = NumberPlane(x_range=[-1, 3], y_range=[-1, 3]).scale(0.5)
        self.place_in_area(plane, 'C3', 'E5', scale_factor=0.8)
        
        b = Vector([2, 1], color=RED)
        a2 = Vector([0, 2], color=BLUE)
        a1 = Vector([1, 0], color=GREEN)
        
        # Vector group on plane
        vectors = VGroup(b, a2, a1)
        self.place_in_area(vectors, 'C3', 'E5', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(colors[0])
        self.play(Create(plane), Create(b), Create(a2))
        self.place_at_grid(grid_icon, 'A6', scale_factor=0.3)
        self.play(FadeIn(grid_icon))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(colors[1])
        # Parallelogram formed by b, a2
        poly = Polygon(ORIGIN, b.get_end(), b.get_end() + a2.get_end(), a2.get_end(), fill_opacity=0.3, color=BLUE)
        self.add(poly)
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(colors[2])
        label = MathTex(r"Area = |det(b, a_2)|").scale(0.8)
        self.place_at_grid(label, 'B5', scale_factor=0.7)
        self.play(Write(label))
        
        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(colors[3])
        ratio = MathTex(r"x_1 = \frac{det(b, a_2)}{det(a_1, a_2)}").scale(0.8)
        self.place_in_area(ratio, 'D4', 'F6', scale_factor=0.9)
        self.place_at_grid(calc_icon, 'F1', scale_factor=0.3)
        self.play(FadeIn(ratio), FadeIn(calc_icon))
        
        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(colors[4])
        self.place_at_grid(prot_icon, 'F6', scale_factor=0.3)
        self.play(Indicate(ratio, color=YELLOW), FadeIn(prot_icon))
        self.wait(2)
