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
        lecture_lines = [
            "Use the Extremal Principle for optimization.",
            "Assume an extreme element forces a contradiction.",
            "Highlights peaks to anchor proofs by induction.",
            "Simpler boundaries aid complex sensor equations.",
            "Finding optimal points."
        ]
        self.setup_layout("Strategy: Thinking in Extremes", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        # Visualization: Number line of extremes (fixed per Issue #27)
        num_line = NumberLine(x_range=[-5, 5, 1], length=6, include_numbers=False)
        extremes = VGroup(Dot(num_line.n2p(-5), color="#FF4500"), Dot(num_line.n2p(5), color="#FF4500"))
        self.place_in_area(num_line, 'D2', 'D5', scale_factor=0.6)
        self.add(num_line, extremes)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFD700"))
        self.play(extremes[0].animate.move_to(num_line.n2p(0)), extremes[1].animate.move_to(num_line.n2p(0)))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        peak = Dot(self.grid['B3'], color="#00FF00", radius=0.2)
        point_label = Text("Peak", font_size=20, color="#00FF00") # Added for Issue #29
        self.place_at_grid(peak, 'B3')
        self.place_at_grid(point_label, 'B3', scale_factor=0.5)
        self.add(peak, point_label)
        self.play(Indicate(peak))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FFD700"))
        # Using SVG Assets per Asset Integration requirement #18
        sensor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sensor.svg")
        boundary = Circle(radius=0.5, color="#00BFFF")
        self.place_at_grid(sensor, 'F3', scale_factor=0.5) # Issue #28 fix
        self.place_at_grid(boundary, 'F3', scale_factor=0.7) # Issue #28 fix
        self.add(sensor, boundary)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FFD700"))
        # Robotic arm indicator asset
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        arm = Line(start=self.grid['F3'], end=self.grid['B3'], color="#FF4500")
        self.place_at_grid(robot, 'F3', scale_factor=0.5)
        self.add(robot, arm)
        self.play(Create(arm))
        self.wait(2)
