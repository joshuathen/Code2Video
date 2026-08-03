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
        title_text = "The Change of Basis Matrix (P)"
        lecture_lines = [
            "This translation is neatly handled by matrix multiplication.",
            "We build matrix P using Pixel's basis vectors.",
            "Each column of P is one of his vectors.",
            "Multiplying P by Pixel's coordinates gives our coordinates.",
            "P acts as a dictionary between these two languages."
        ]
        self.setup_layout(title_text, lecture_lines)

        def highlight_line(index):
            return [self.lecture[i].animate.set_color(YELLOW if i == index else WHITE) for i in range(len(self.lecture))]

        # Colors
        COLOR_V1 = "#FFFF00" # Yellow
        COLOR_V2 = "#00FFFF" # Teal

        # === Animation for Lecture Line 1 ===
        self.play(*highlight_line(0))
        
        # Abstract matrix P
        matrix_p_raw = MathTex(r"P = \begin{bmatrix} p_{11} & p_{12} \\ p_{21} & p_{22} \end{bmatrix}", font_size=42)
        self.place_in_area(matrix_p_raw, "C3", "D5")
        self.play(Write(matrix_p_raw))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        self.play(*highlight_line(1))
        
        v1_tex = MathTex(r"\vec{b}_1 = \begin{bmatrix} 2 \\ 1 \end{bmatrix}", color=COLOR_V1, font_size=36)
        v2_tex = MathTex(r"\vec{b}_2 = \begin{bmatrix} -1 \\ 1 \end{bmatrix}", color=COLOR_V2, font_size=36)
        
        # Applied fix for Issue 44: v1_tex positioned at B4
        self.place_at_grid(v1_tex, 'B4')
        self.place_at_grid(v2_tex, "B5")
        
        self.play(FadeIn(v1_tex), FadeIn(v2_tex))
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        self.play(*highlight_line(2))
        
        # Numeric matrix P
        matrix_p_final = MathTex(r"P = \begin{bmatrix} 2 & -1 \\ 1 & 1 \end{bmatrix}", font_size=42)
        self.place_in_area(matrix_p_final, "C3", "D5")
        
        # Color columns based on content
        matrix_p_final.set_color_by_tex("2", COLOR_V1)
        matrix_p_final.set_color_by_tex("-1", COLOR_V2)
        ones = matrix_p_final.get_parts_by_tex("1")
        if len(ones) >= 2:
            ones[0].set_color(COLOR_V1) # bottom left
            ones[1].set_color(COLOR_V2) # bottom right

        self.play(
            ReplacementTransform(matrix_p_raw, matrix_p_final),
            v1_tex.animate.scale(0.5).move_to(matrix_p_final.get_center()).set_opacity(0),
            v2_tex.animate.scale(0.5).move_to(matrix_p_final.get_center()).set_opacity(0),
        )
        self.play(Indicate(matrix_p_final))
        self.wait(1.5)

        # === Animation for Lecture Line 4 ===
        self.play(*highlight_line(3))
        
        equation = MathTex(r"\vec{x} = P [\vec{x}]_B", font_size=40)
        self.place_at_grid(equation, "A4")
        
        # Example calculation
        calc_example = MathTex(
            r"\begin{bmatrix} 3 \\ 3 \end{bmatrix} = \begin{bmatrix} 2 & -1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 2 \\ 1 \end{bmatrix}",
            font_size=32
        )
        # Applied fix for Issue 45: calc_example in larger area E2-F6
        self.place_in_area(calc_example, 'E2', 'F6')
        
        # Color matrix part for visual link
        calc_example.set_color_by_tex("2", COLOR_V1)
        calc_example.set_color_by_tex("-1", COLOR_V2)
        
        self.play(Write(equation))
        self.play(FadeIn(calc_example, shift=UP * 0.3))
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        self.play(*highlight_line(4))
        
        dict_label = Text("Dictionary Matrix", font_size=24, color=YELLOW)
        # Applied fix for Issue 46: dict_label scale 0.6 at D6
        self.place_at_grid(dict_label, 'D6', scale_factor=0.6)
        arrow = Arrow(dict_label.get_left(), matrix_p_final.get_right(), color=YELLOW, buff=0.1)
        
        self.play(Create(arrow), Write(dict_label))
        self.wait(3)
