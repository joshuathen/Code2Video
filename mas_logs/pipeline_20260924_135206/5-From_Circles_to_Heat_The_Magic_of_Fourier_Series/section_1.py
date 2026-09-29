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
        self.setup_layout("Prerequisites: The Circular Motion Foundation", 
                          ["Euler's formula describes periodic rotation.", 
                           "Periodic functions are sums of rotating vectors.", 
                           "These combined rotations create complex shapes."])
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        circle = Circle(radius=1.5, color=WHITE)
        self.place_at_grid(circle, 'C4', scale_factor=0.9)
        label_circle = Text("Circular Path", font_size=16, color=WHITE)
        self.place_at_grid(label_circle, 'B4', scale_factor=0.8)
        self.play(Create(circle), Write(label_circle))

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        vector = Line(start=circle.get_center(), end=circle.get_center() + RIGHT * 1.5, color="#FF00FF")
        label_vector = Text("Euler Vector", font_size=16, color="#FF00FF")
        self.place_at_grid(label_vector, 'D5', scale_factor=0.7)
        self.play(Create(vector), Write(label_vector))

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        # Animate rotation
        self.play(Rotate(vector, angle=2*PI, about_point=circle.get_center(), run_time=3, rate_func=linear))
        self.wait(1)
