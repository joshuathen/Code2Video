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
        self.setup_layout("Dynamic Notation Mapping", ["All forms are unified here.", "Two cubed is eight.", "Log base two eight is three."])
        
        # Mobjects
        eq_exp = MathTex(r"2^3 = 8", color=YELLOW)
        eq_root = MathTex(r"\sqrt[3]{8} = 2", color=TEAL)
        eq_log = MathTex(r"\log_{2}(8) = 3", color=GREEN)
        
        label_1 = Text("Exponential", font_size=24, color=YELLOW)
        label_2 = Text("Root", font_size=24, color=TEAL)
        label_3 = Text("Logarithmic", font_size=24, color=GREEN)
        
        cube_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg", color=TEAL_C)

        # Asset integrated visualization
        self.place_at_grid(cube_icon, 'A3', scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.place_at_grid(eq_exp, 'B3', scale_factor=0.9)
        self.place_at_grid(label_1, 'B4', scale_factor=0.7)
        self.play(Write(eq_exp), FadeIn(label_1), FadeIn(cube_icon))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(TEAL)
        self.place_at_grid(eq_root, 'C3', scale_factor=0.9)
        self.place_at_grid(label_2, 'C4', scale_factor=0.7)
        self.play(TransformFromCopy(eq_exp, eq_root), FadeIn(label_2))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(GREEN)
        self.place_at_grid(eq_log, 'D3', scale_factor=0.9)
        self.place_at_grid(label_3, 'D4', scale_factor=0.7)
        self.play(TransformFromCopy(eq_exp, eq_log), FadeIn(label_3))
        
        # Grouping for visual balance
        group_all_formulas = VGroup(eq_exp, eq_root, eq_log, label_1, label_2, label_3)
        self.place_in_area(group_all_formulas, 'B3', 'D5', scale_factor=0.8)

        self.wait(2)
