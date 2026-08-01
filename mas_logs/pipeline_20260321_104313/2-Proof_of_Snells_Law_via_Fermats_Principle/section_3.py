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
        # Initialize lecture lines
        lecture_lines = [
            "Let point A be in air and B in water.",
            "Light hits the interface at a variable point x.",
            "Vertical heights are a and b; horizontal distance is d.",
            "Path L1 connects point A to the interface.",
            "Path L2 connects the interface to point B."
        ]
        self.setup_layout("Geometric Setup of the Boundary", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Let point A be in air and B in water.
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        
        # Media visualization (Air and Water)
        # Using A1 to C6 for Air and C1 to F6 for Water area
        air_rect = Rectangle(width=5.0, height=2.0, fill_opacity=0.2, fill_color=BLUE_A, stroke_width=0)
        self.place_in_area(air_rect, 'A1', 'C6')
        
        water_rect = Rectangle(width=5.0, height=3.0, fill_opacity=0.2, fill_color=BLUE_E, stroke_width=0)
        self.place_in_area(water_rect, 'C1', 'F6')
        
        # Interface line (horizontal through row C)
        interface = Line(self.grid['C1'], self.grid['C6'], color=WHITE)
        
        # Point A at B2 and Point B at E5
        dot_a = Dot(color="#FFFF00")
        self.place_at_grid(dot_a, 'B2')
        label_a_pt = Text("A", font_size=24, color="#FFFF00").next_to(dot_a, UP, buff=0.1)
        
        dot_b = Dot(color="#FF00FF")
        self.place_at_grid(dot_b, 'E5')
        label_b_pt = Text("B", font_size=24, color="#FF00FF").next_to(dot_b, DOWN, buff=0.1)
        
        self.play(Create(air_rect), Create(water_rect))
        self.play(Create(interface))
        self.play(FadeIn(dot_a), FadeIn(label_a_pt), FadeIn(dot_b), FadeIn(label_b_pt))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Light hits the interface at a variable point x.
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        
        # Point P on the interface (C4)
        dot_p = Dot(color="#00FFFF")
        self.place_at_grid(dot_p, 'C4')
        label_p = Text("P", font_size=24, color="#00FFFF").next_to(dot_p, DOWN, buff=0.1)
        
        # Horizontal distance 'x' (from projection of A at C2 to P at C4)
        # We define a visual indicator for x
        x_line = Line(self.grid['C2'], self.grid['C4'], color="#00FFFF").shift(DOWN * 0.4)
        label_x = Text("x", font_size=20, color="#00FFFF").next_to(x_line, DOWN, buff=0.05)
        
        self.play(FadeIn(dot_p), FadeIn(label_p))
        self.play(Create(x_line), FadeIn(label_x))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Vertical heights are a and b; horizontal distance is d.
        self.play(self.lecture[2].animate.set_color(WHITE))
        
        # Vertical heights a and b
        dash_a = DashedLine(self.grid['B2'], self.grid['C2'], color="#FFFF00")
        label_a_height = Text("a", font_size=20, color="#FFFF00").next_to(dash_a, LEFT, buff=0.1)
        
        dash_b = DashedLine(self.grid['E5'], self.grid['C5'], color="#FF00FF")
        label_b_height = Text("b", font_size=20, color="#FF00FF").next_to(dash_b, RIGHT, buff=0.1)
        
        # Total horizontal distance d (from projection of A at C2 to projection of B at C5)
        d_line = Line(self.grid['C2'], self.grid['C5'], color=WHITE).shift(DOWN * 1.0)
        label_d = Text("d", font_size=20, color=WHITE).next_to(d_line, DOWN, buff=0.05)
        
        self.play(Create(dash_a), FadeIn(label_a_height))
        self.play(Create(dash_b), FadeIn(label_b_height))
        self.play(Create(d_line), FadeIn(label_d))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Path L1 connects point A to the interface.
        self.play(self.lecture[3].animate.set_color("#FFFF00"))
        
        l1_path = Line(self.grid['B2'], self.grid['C4'], color="#FFFF00")
        label_l1 = Text("L1", font_size=22, color="#FFFF00").next_to(l1_path.get_center(), UP + LEFT, buff=0.1)
        
        self.play(Create(l1_path), FadeIn(label_l1))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Path L2 connects the interface to point B.
        self.play(self.lecture[4].animate.set_color("#FF00FF"))
        
        l2_path = Line(self.grid['C4'], self.grid['E5'], color="#FF00FF")
        label_l2 = Text("L2", font_size=22, color="#FF00FF").next_to(l2_path.get_center(), DOWN + RIGHT, buff=0.1)
        
        self.play(Create(l2_path), FadeIn(label_l2))
        self.wait(2)
