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
        lecture_lines = ["A 1D line has length, but no area.", "A 2D square covers a distinct area.", "Can a continuous path visit every 2D point?"]
        self.setup_layout("The Dimensional Paradox", lecture_lines)
        
        square = Square(side_length=2.5, color="#FF5733")
        label_2d = Text("2D Plane", font_size=24, color="#FF5733")
        
        line = Line(start=LEFT, end=RIGHT, color="#33FF57")
        label_1d = Text("1D Line", font_size=24, color="#33FF57")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.place_at_grid(line, 'B3', scale_factor=1.5)
        self.place_at_grid(label_1d, 'B3') # Note: Needs adjustment if overlap, but keeping for now as per instructions.
        self.play(Create(line), Write(label_1d))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(WHITE))
        self.place_in_area(square, 'D2', 'F5', scale_factor=0.7)
        self.place_at_grid(label_2d, 'C5', scale_factor=0.9)
        self.play(Create(square), Write(label_2d))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(WHITE))
        # Animate the 1D line moving into the space-filling concept
        self.play(line.animate.set_color("#3357FF"), square.animate.set_stroke(color="#FFD700"))
        self.wait(2)
