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
        lecture_lines = ["Pi links diameter to circumference.", "It transcends time and scale.", "Pi is a fundamental geometric bridge."]
        self.setup_layout("Summary and Conclusion", lecture_lines)
        
        # Define visual elements
        circle = Circle(radius=1, color=WHITE)
        diameter = Line(circle.get_left(), circle.get_right(), color=RED)
        label_pi = MathTex(r"\pi = C/D", color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        # Positioned using area as suggested
        self.place_in_area(circle, 'B1', 'C3', scale_factor=0.6)
        self.place_in_area(diameter, 'B1', 'C3', scale_factor=0.6)
        # Positioned at B5 as suggested
        self.place_at_grid(label_pi, 'B5', scale_factor=0.9)
        
        self.play(Create(circle), Create(diameter), Write(label_pi))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Visualizing time/history as a timeline on right grid
        history_line = Line(self.grid["D1"], self.grid["D6"], color=YELLOW)
        point1 = Dot(self.grid["D1"], color=YELLOW)
        point2 = Dot(self.grid["D6"], color=YELLOW)
        self.play(Create(history_line), GrowFromCenter(point1), GrowFromCenter(point2))
        
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Final visualization positioned at D2 as suggested
        pi_symbol = MathTex(r"\pi", font_size=144, color="#FF00FF")
        self.place_at_grid(pi_symbol, 'D2', scale_factor=1.0)
        self.play(Write(pi_symbol))
        self.play(self.lecture[2].animate.set_color("#FF00FF"))
        self.wait(2)
