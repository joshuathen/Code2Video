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
        # Setup layout with mandatory script lines
        lines = [
            'Look back at our original area of five.',
            'Swap the first vector V1 for target vector B.',
            'This swap creates a brand new green parallelogram.',
            'The new area measures exactly ten units.',
            'The ratio of areas gives us the solution x.'
        ]
        self.setup_layout("The Geometric Trick: Replacing a Column", lines)

        # Define Geometric Elements
        origin_point = self.grid['E2']
        
        # Vectors relative to the chosen origin
        v1_vec = np.array([1.5, 0, 0])
        v2_vec = np.array([0.5, 1.5, 0])
        b_vec = np.array([3.0, 0, 0]) # x=2 relative to v1
        
        # Manim Objects
        v1 = Arrow(origin_point, origin_point + v1_vec, buff=0, color="#0000FF")
        v2 = Arrow(origin_point, origin_point + v2_vec, buff=0, color="#FF0000")
        b = Arrow(origin_point, origin_point + b_vec, buff=0, color="#00FF00")
        
        v1_label = Text("V1", color="#0000FF").scale(0.6)
        v2_label = Text("V2", color="#FF0000").scale(0.6)
        b_label = Text("B", color="#00FF00").scale(0.6)
        
        # Positioning labels using grid logic (addressing issues 37 and 38)
        self.place_at_grid(v1_label, 'F4')
        self.place_at_grid(v2_label, 'D2')
        self.place_at_grid(b_label, 'F4')
        
        # Parallelograms
        poly_orig = Polygon(
            origin_point, 
            origin_point + v1_vec, 
            origin_point + v1_vec + v2_vec, 
            origin_point + v2_vec, 
            fill_opacity=0.4, 
            fill_color="#FFFF00", 
            stroke_color="#FFFF00"
        )
        
        poly_new = Polygon(
            origin_point, 
            origin_point + b_vec, 
            origin_point + b_vec + v2_vec, 
            origin_point + v2_vec, 
            fill_opacity=0.4, 
            fill_color="#00FF00", 
            stroke_color="#00FF00"
        )
        
        area_label_orig = Text("Area = 5", color="#FFFF00").scale(0.6)
        area_label_new = Text("Area = 10", color="#00FF00").scale(0.6)
        
        # Positioning area labels (addressing issue 39)
        self.place_at_grid(area_label_orig, 'C4')
        self.place_at_grid(area_label_new, 'B4')

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW), run_time=0.5)
        self.add(v1, v2, v1_label, v2_label, poly_orig)
        self.play(FadeIn(area_label_orig))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(YELLOW), run_time=0.5)
        self.play(
            ReplacementTransform(v1, b),
            ReplacementTransform(v1_label, b_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(YELLOW), run_time=0.5)
        self.play(
            FadeOut(poly_orig),
            FadeOut(area_label_orig),
            Create(poly_new)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[2].animate.set_color(WHITE), self.lecture[3].animate.set_color(YELLOW), run_time=0.5)
        self.play(FadeIn(area_label_new))
        self.wait(1.5)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[3].animate.set_color(WHITE), self.lecture[4].animate.set_color(YELLOW), run_time=0.5)
        
        eq1 = Text("x = 10 / 5", color=WHITE).scale(0.8)
        eq2 = Text("x = 2", color=WHITE).scale(1.0)
        
        self.place_at_grid(eq1, 'A3')
        self.place_at_grid(eq2, 'A3')
        
        self.play(Write(eq1))
        self.wait(1)
        self.play(ReplacementTransform(eq1, eq2))
        self.wait(2)
