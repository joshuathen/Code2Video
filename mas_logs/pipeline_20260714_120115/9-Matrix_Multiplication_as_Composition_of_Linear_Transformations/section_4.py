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
        # Data from storyboard: "The Right-to-Left Rule"
        title = "The Right-to-Left Rule"
        lecture_lines = [
            "In BA, matrix A is applied first.",
            "We read the order from right to left.",
            "This matches how nested functions like B(A(v)) work."
        ]
        
        self.setup_layout(title, lecture_lines)
        
        # Define colors as per storyboard/instruction
        color_B = "#ADD8E6" # Light Blue
        color_A = "#FFFFE0" # Light Yellow
        color_v = "#FFFFFF" # White
        highlight_color = YELLOW
        
        # === Animation for Lecture Line 1 ===
        # Line 1: "In BA, matrix A is applied first."
        self.next_section("Notation")
        self.lecture[0].set_color(highlight_color)
        
        tex_B = Text("B", color=color_B)
        tex_dot1 = Text("·")
        tex_A = Text("A", color=color_A)
        tex_dot2 = Text("·")
        tex_v = Text("v", color=color_v, weight=BOLD)
        
        formula_bav = VGroup(tex_B, tex_dot1, tex_A, tex_dot2, tex_v).arrange(RIGHT, buff=0.1)
        # Positioned in Row B to avoid empty space at top (Issue 37) and scaled to 1.2 (Issue 38)
        self.place_in_area(formula_bav, "B2", "B5", scale_factor=1.2)
        
        self.play(FadeIn(formula_bav))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Line 2: "We read the order from right to left."
        self.next_section("Processing Order")
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(highlight_color)
        
        # Pulse right-to-left order: v -> A -> B
        self.play(Indicate(tex_v, color=color_v, scale_factor=1.3))
        self.play(Indicate(tex_A, color=color_A, scale_factor=1.3))
        self.play(Indicate(tex_B, color=color_B, scale_factor=1.3))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Line 3: "This matches how nested functions like B(A(v)) work."
        self.next_section("Functional Analogy")
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(highlight_color)
        
        # Create nested function text B(A(v))
        tex_f_B = Text("B(", color=color_B)
        tex_f_A = Text("A(", color=color_A)
        tex_f_v = Text("v", color=color_v, weight=BOLD)
        tex_f_p1 = Text(")")
        tex_f_p2 = Text(")")
        
        nested_formula = VGroup(tex_f_B, tex_f_A, tex_f_v, tex_f_p1, tex_f_p2).arrange(RIGHT, buff=0.05)
        # Positioned in Row D for balance (Issue 36) and scaled to 1.2 (Issue 38)
        self.place_in_area(nested_formula, "D2", "D5", scale_factor=1.2)
        
        # Flow arrow indicating right-to-left
        flow_arrow = Arrow(
            tex_v.get_top() + UP * 0.2,
            tex_B.get_top() + UP * 0.2,
            path_arc=-1.2,
            color=WHITE,
            buff=0.1
        )
        
        self.play(
            Write(nested_formula),
            Create(flow_arrow)
        )
        self.wait(3)
