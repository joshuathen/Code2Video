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
        lecture_lines = ["Dimensions define our degrees of spatial freedom.", "A 2D Flatlander cannot perceive 3D depth.", "We appear strange to lower-dimensional beings."]
        self.setup_layout("Introduction: The Flatlander Paradox", lecture_lines)
        
        # 0D Point
        dot = Dot(color=WHITE)
        label_dot = Text("Point", color=WHITE, font_size=20)
        self.place_at_grid(dot, 'B2', scale_factor=0.8)
        label_dot.next_to(dot, DOWN)
        
        # 1D Line
        line = Line(start=self.grid['B3'], end=self.grid['B5'], color="#FF00FF")
        label_line = Text("Line", color="#FF00FF", font_size=20)
        self.place_at_grid(label_line, 'B4', scale_factor=0.9)
        
        # 2D Square
        square = Square(side_length=0.8, color="#00FFFF")
        self.place_at_grid(square, 'D2', scale_factor=0.8)
        label_square = Text("Square", color="#00FFFF", font_size=20)
        self.place_at_grid(label_square, 'E2', scale_factor=0.7)
        
        # 3D Cube
        cube = Cube(side_length=0.8, fill_opacity=0.5, color="#FFFF00")
        self.place_at_grid(cube, 'D5', scale_factor=0.8)
        label_cube = Text("Cube", color="#FFFF00", font_size=20)
        self.place_at_grid(label_cube, 'E5', scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF99"))
        self.play(Create(dot), Write(label_dot))
        self.play(Create(line), Write(label_line))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#99FFFF"))
        self.play(Create(square), Write(label_square))
        self.play(Rotate(square, angle=PI/4))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF99FF"))
        self.play(Create(cube), Write(label_cube))
        self.play(Rotate(cube, angle=PI/4, axis=UP + RIGHT))
        self.wait(2)
