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
        self.setup_layout("The Conversion Algorithm", [
            "Standard space equals P times new.",
            "Use inverse P to reverse.",
            "Visualize grid un-warping clearly.",
            "Reverse operation shows system reversibility.",
            "System conversion is fully computable."
        ])
        
        # Assets
        matrix_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/matrix.svg")
        calc_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg")
        
        # === Animation for Lecture Line 1 ===
        # Standard space equals P times new.
        formula1 = MathTex(r"[v]_{std} = P \cdot [v]_{new}", font_size=36)
        self.place_in_area(formula1, 'A2', 'B5', scale_factor=0.9)
        self.play(Write(formula1))
        self.lecture[0].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Use inverse P to reverse.
        formula2 = MathTex(r"[v]_{new} = P^{-1} \cdot [v]_{std}", font_size=36)
        self.place_at_grid(formula2, 'B5', scale_factor=0.8)
        
        # Highlight P^-1 with matrix icon
        self.place_at_grid(matrix_icon, 'C5', scale_factor=0.3)
        self.play(FadeIn(matrix_icon), Write(formula2))
        
        self.lecture[1].set_color(BLUE)
        p_inv = formula2[0][5:8]
        self.play(Indicate(p_inv, color="#FF69B4"))
        self.lecture[1].set_color("#FF69B4")
        self.wait(1)

        # === Animation for Lecture Line 3, 4, 5 ===
        visual_text = Text("Un-warping grid visual", font_size=18, color=GRAY)
        rev_text = Text("Reversibility confirmed", font_size=18, color=WHITE)
        comp_text = Text("Computation Ready", font_size=18, color=WHITE)
        group_of_texts = VGroup(visual_text, rev_text, comp_text).arrange(DOWN, aligned_edge=LEFT)
        
        self.place_in_area(group_of_texts, 'C2', 'E5', scale_factor=0.75)
        self.place_at_grid(calc_icon, 'F5', scale_factor=0.3)

        self.play(FadeIn(group_of_texts))
        self.lecture[2].set_color(GREEN)
        self.wait(0.5)
        self.lecture[3].set_color(ORANGE)
        self.wait(0.5)
        self.lecture[4].set_color(PURPLE)
        self.play(FadeIn(calc_icon))
        self.wait(2)
