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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Determinants calculate the signed area of a parallelogram.",
            "In 2D, these vectors span a simple flat plane.",
            "This area is defined by the formula ad minus bc."
        ]
        self.setup_layout("Prerequisite Review: Vectors and Determinants", lecture_lines)
        
        # Assets
        parallelogram_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/parallelogram.svg")
        
        # Elements
        v_a = Vector([1, 1], color="#FF5733")
        v_b = Vector([1, -0.5], color="#33FF57")
        origin = ORIGIN
        
        # Parallelgram area
        parallelogram_shape = Polygon(origin, v_a.get_end(), v_a.get_end() + v_b.get_end(), v_b.get_end(), color="#33A1FF", fill_opacity=0.3)
        formula = MathTex(r"|A, B| = a_x b_y - a_y b_x", font_size=36, color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        # Placing vectors
        self.place_at_grid(v_a, 'C3', scale_factor=0.8)
        self.place_at_grid(v_b, 'C5', scale_factor=0.8)
        self.play(Create(v_a), Create(v_b))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color("#FFD700"))
        # Using asset
        self.place_at_grid(parallelogram_icon, 'C4', scale_factor=0.5)
        self.play(FadeIn(parallelogram_icon))
        
        # Fix 23: Parallelogram visualization
        self.place_at_grid(parallelogram_shape, 'C3', scale_factor=0.9)
        self.play(Create(parallelogram_shape))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color("#FFD700"))
        # Fix 22: formula placement
        self.place_in_area(formula, 'E1', 'F6', scale_factor=0.6)
        self.play(Write(formula))
        
        self.wait(2)
