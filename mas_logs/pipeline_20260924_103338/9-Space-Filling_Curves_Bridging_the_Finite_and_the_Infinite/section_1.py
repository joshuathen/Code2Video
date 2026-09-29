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
        lecture_lines = ["A line is one-dimensional, having zero area.", "Can a one-dimensional path fill a two-dimensional square?", "Imagine an ant trying to touch every square point."]
        self.setup_layout("The Intuitive Paradox: Dimension vs. Path", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        square = Square(side_length=2, color=WHITE)
        # Apply Fix 17
        self.place_at_grid(square, 'C3', scale_factor=1.2)
        label = Text("1D vs 2D", font_size=24, color=WHITE)
        # Apply Fix 18
        self.place_at_grid(label, 'B3', scale_factor=1.0)
        self.play(Create(square), Write(label))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        # Load asset and path
        ant = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ant.svg")
        # Apply Fix 19
        self.place_in_area(ant, 'B4', 'F6', scale_factor=0.9)
        
        line = Line(start=square.get_left(), end=square.get_right(), color=YELLOW)
        self.play(ReplacementTransform(ant, line))
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        fill_path = VMobject(color="#FF5733")
        fill_path.set_points_smoothly([square.get_corner(UL), square.get_corner(UR), square.get_corner(DL), square.get_corner(DR)])
        ant_dot = Dot(color=RED)
        ant_dot.move_to(square.get_corner(UL))
        self.play(MoveAlongPath(ant_dot, fill_path), run_time=3)
        self.lecture[2].set_color("#FF5733")
        self.wait(1)
