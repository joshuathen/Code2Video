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
            "The Hilbert curve is highly efficient.",
            "It preserves locality between nearby points.",
            "Think of a robot cleaning a room.",
            "It never crosses its own track.",
            "Total travel distance is minimized here."
        ]
        self.setup_layout("The Hilbert Curve: Efficiency in Data Mapping", lecture_lines)
        
        # Load asset
        robot_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg"
        robot = SVGMobject(robot_asset)
        
        # Grid setup (representing data space)
        grid = VGroup(*[Square(side_length=0.7, stroke_color=WHITE, stroke_width=2) for _ in range(4)])
        grid.arrange_in_grid(2, 2, buff=0)
        # Applying critique fixes for position
        self.place_in_area(grid, 'B2', 'F6', scale_factor=1.1)
        
        # === Animation for Lecture Line 1 ===
        # 1. Present a 2x2 grid representing data space (#FFFFFF). [Asset: ...robot.svg]
        self.play(FadeIn(grid))
        self.place_at_grid(robot, "B3", scale_factor=0.3)
        self.play(FadeIn(robot))
        self.lecture[0].set_color("#FFD700")

        # === Animation for Lecture Line 2 ===
        # 2. Draw Hilbert curve connecting cells (#FF0000).
        points = [grid[0].get_center(), grid[1].get_center(), grid[3].get_center(), grid[2].get_center()]
        path = VMobject(color="#FF0000", stroke_width=4)
        path.set_points_smoothly([p for p in points])
        self.play(Create(path))
        self.lecture[1].set_color("#00FF00")

        # === Animation for Lecture Line 3 ===
        # 3. Increase recursion level, highlight locality (#00FF00). [Asset: ...robot.svg]
        self.lecture[2].set_color("#00BFFF")
        self.play(robot.animate.move_to(path.get_end()))

        # === Animation for Lecture Line 4 ===
        self.play(MoveAlongPath(robot, path), run_time=2)
        self.lecture[3].set_color("#FF69B4")

        # === Animation for Lecture Line 5 ===
        self.play(Indicate(path, color="#FFFFFF"))
        self.lecture[4].set_color("#FF4500")
        self.wait(1)
