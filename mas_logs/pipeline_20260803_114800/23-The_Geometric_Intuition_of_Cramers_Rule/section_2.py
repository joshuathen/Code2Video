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
        lecture_lines = [
            "- The determinant represents the area of a parallelogram.",
            "- Vectors A and B form our system's base area.",
            "- This area is the determinant of matrix AB."
        ]
        self.setup_layout("Prerequisite: Determinant as Area", lecture_lines)
        
        # Define Vectors A and B relative to a local origin
        v_a_dir = np.array([2, 0, 0])
        v_b_dir = np.array([0.5, 1.5, 0])
        
        vec_a = Arrow(start=ORIGIN, end=v_a_dir, buff=0, color="#00FF00")
        vec_b = Arrow(start=ORIGIN, end=v_b_dir, buff=0, color="#0000FF")
        
        # Using simplified MathTex and applying color via set_color to avoid TeX template issues
        label_a = MathTex(r"\vec{A}").set_color("#00FF00").scale(0.8)
        label_b = MathTex(r"\vec{B}").set_color("#0000FF").scale(0.8)
        
        # Parallelogram outline
        poly = Polygon(
            ORIGIN, v_a_dir, v_a_dir + v_b_dir, v_b_dir,
            stroke_width=2, stroke_color=WHITE
        )
        # Parallelogram fill
        poly_fill = poly.copy().set_fill("#888888", opacity=0.5).set_stroke(width=0)
        
        # Area label using \det for robust TeX compilation
        area_label = MathTex(r"\det(A, B)").set_color(WHITE).scale(0.8)
        
        # Internal positioning relative to the vectors/polygon before grouping
        label_a.next_to(vec_a.get_end(), DOWN, buff=0.1)
        label_b.next_to(vec_b.get_end(), LEFT, buff=0.1)
        area_label.move_to(poly.get_center())
        
        # Group components for initial positioning
        viz_group = VGroup(poly_fill, poly, vec_a, vec_b, label_a, label_b, area_label)
        
        # Place the whole group in the area B3 to E6
        self.place_in_area(viz_group, "B3", "E6", scale_factor=1.0)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(
            Create(vec_a), 
            Create(vec_b), 
            Write(label_a), 
            Write(label_b),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.play(Create(poly), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        self.play(
            FadeIn(poly_fill), 
            Write(area_label),
            run_time=1.5
        )
        self.wait(2)
        
        # Finish
        self.lecture[2].set_color(WHITE)
        self.wait(1)
