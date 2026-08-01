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

class Section6Scene(TeachingScene):
    def construct(self):
        # Title and Lecture Lines
        title = "Summary: The Dimensional Toolkit"
        lines = [
            "- Use dimensional reduction to simplify paths and distances.",
            "- Use dimensional expansion to bypass seemingly solid obstacles.",
            "- When stuck, ask if you are in the right dimension."
        ]
        self.setup_layout(title, lines)
        
        # Colors for the toolkit
        REDUCTION_COLOR = BLUE
        EXPANSION_COLOR = GREEN
        STUCK_COLOR = YELLOW
        
        # === Animation for Lecture Line 1 ===
        # Icons for 'Reduction' and 'Expansion' appear on a split screen.
        self.play(self.lecture[0].animate.set_color(REDUCTION_COLOR))
        
        # Split screen divider - positioned between columns 3 and 4
        p_top = (self.grid["A3"] + self.grid["A4"]) / 2
        p_bottom = (self.grid["F3"] + self.grid["F4"]) / 2
        split_line = Line(p_top, p_bottom, color=GRAY_E)
        
        # Reduction Icon: Representing reduction (3D to 2D)
        reduction_sq1 = Square(side_length=1.2, color=REDUCTION_COLOR)
        reduction_sq2 = Square(side_length=1.2, color=REDUCTION_COLOR).shift(UP*0.3 + RIGHT*0.3)
        reduction_lines = VGroup(*[
            Line(reduction_sq1.get_vertices()[i], reduction_sq2.get_vertices()[i], color=REDUCTION_COLOR)
            for i in range(4)
        ])
        reduction_icon = VGroup(reduction_sq1, reduction_sq2, reduction_lines).scale(0.5)
        reduction_label = Text("Reduction", font_size=20, color=REDUCTION_COLOR)
        self.reduction_group = VGroup(reduction_icon, reduction_label).arrange(DOWN, buff=0.4)
        
        # Expansion Icon: Representing expansion (Point to Circle/Disk)
        expansion_point = Dot(color=EXPANSION_COLOR)
        expansion_circle = Circle(radius=0.6, color=EXPANSION_COLOR).set_opacity(0.3)
        expansion_icon = VGroup(expansion_point, expansion_circle).scale(0.5)
        expansion_label = Text("Expansion", font_size=20, color=EXPANSION_COLOR)
        self.expansion_group = VGroup(expansion_icon, expansion_label).arrange(DOWN, buff=0.4)
        
        # Positioning according to Issue 65
        self.place_in_area(self.reduction_group, 'A2', 'C3', scale_factor=0.8)
        self.place_in_area(self.expansion_group, 'A4', 'C5', scale_factor=0.8)
        
        self.play(Create(split_line), FadeIn(self.reduction_group), FadeIn(self.expansion_group))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # A white #FFFFFF 2D [Asset: maze.svg] appears with a character [Asset: dot.svg].
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(EXPANSION_COLOR)
        )
        
        # Assets loading
        maze = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/maze.svg").set_color(WHITE)
        self.char_dot = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/dot.svg").set_color(RED)
        
        # Layout: Maze in the central/lower grid area
        self.place_in_area(maze, "B2", "F6", scale_factor=0.7)
        # Position dot inside the maze - centering according to Issue 65
        self.place_at_grid(self.char_dot, 'D4', scale_factor=0.3)
        
        self.play(
            FadeOut(self.reduction_group),
            FadeOut(self.expansion_group),
            FadeOut(split_line),
            FadeIn(maze),
            FadeIn(self.char_dot)
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # The [Asset: dot.svg] dot scales up and 'jumps' over walls to the exit.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(STUCK_COLOR)
        )
        
        # Define exit point at the bottom right of the maze
        exit_point = self.grid["F5"]
        
        # Dimensional Expansion: Jumping out of the 2D plane
        # 1. Scale up and change color to represent 'ascending'
        self.play(
            self.char_dot.animate.scale(3).set_color(ORANGE).set_opacity(0.8),
            run_time=1
        )
        # 2. Move across the 'flat' maze barriers
        self.play(
            self.char_dot.animate.move_to(exit_point),
            run_time=2,
            rate_func=slow_into
        )
        # 3. Scale back down and return color to represent 'descending'
        self.play(
            self.char_dot.animate.scale(1/3).set_color(RED).set_opacity(1),
            run_time=1
        )
        
        self.wait(3)

# Mark issues as resolved
# update_issue(65, under_review=True, resolution_note="Integrated SVGMobjects for maze and dot. Adjusted layout for reduction and expansion groups and centered the character dot as requested. Ensured 1:1 timing with lecture lines.")
