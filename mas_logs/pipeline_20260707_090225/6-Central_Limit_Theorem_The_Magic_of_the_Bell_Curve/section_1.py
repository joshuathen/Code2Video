from manim import *
import random

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
        # Setup the layout
        title = "The Mystery of Predictable Chaos"
        lines = [
            "Welcome to the world of unpredictable and chaotic data.",
            "Imagine a forest filled with giant, oddly weighted squirrels.",
            "Individual weights are messy, following no specific pattern."
        ]
        self.setup_layout(title, lines)
        
        # Asset path
        SQUIRREL_ASSET = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/squirrel.svg"
        SQUIRREL_COLOR = "#FFFFFF"
        HIGHLIGHT_COLOR = "#FFFF00"
        
        # === Animation for Lecture Line 1 ===
        # Create a scatter of 15 squirrel icons with varying sizes (1kg to 20kg)
        squirrels = VGroup()
        for i in range(15):
            # Weight determines the relative scale of the individual SVG
            weight = random.uniform(1, 20)
            sq_scale = 0.05 + (weight / 20) * 0.15
            sq = SVGMobject(SQUIRREL_ASSET)
            sq.set_color(SQUIRREL_COLOR)
            sq.scale(sq_scale)
            # Assign random local positions before place_in_area centers the group
            sq.move_to([random.uniform(-2, 2), random.uniform(-1, 1), 0])
            squirrels.add(sq)
        
        # Layout Fix (Squirrels): Center the scatter in the specified area
        self.place_in_area(squirrels, 'B1', 'C6', scale_factor=0.6)
            
        self.play(
            self.lecture[0].animate.set_color(SQUIRREL_COLOR),
            FadeIn(squirrels, shift=UP),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Jiggle squirrels to represent chaotic nature
        self.play(
            self.lecture[1].animate.set_color(SQUIRREL_COLOR),
            *[sq.animate.shift(np.array([random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), 0])) for sq in squirrels],
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Create a horizontal axis
        axis = NumberLine(
            x_range=[0, 20, 5],
            length=5,
            color=WHITE,
            include_numbers=True,
            font_size=18,
            label_constructor=Text
        )
        # Layout Fix (Axis): Move axis higher to avoid cut-offs
        self.place_in_area(axis, 'E1', 'E6', scale_factor=1.0)
        
        # Transition the squirrels into a flat histogram (uniform distribution along axis)
        target_positions = []
        axis_start = axis.n2p(0)
        axis_end = axis.n2p(20)
        
        for i in range(15):
            # Spread linearly to show no normality
            pos_x = axis_start[0] + (i / 14) * (axis_end[0] - axis_start[0])
            target_positions.append(np.array([pos_x, axis_start[1] + 0.35, 0]))

        # Highlight a single squirrel icon to emphasize unpredictability
        target_squirrel = squirrels[7]
        question_mark = Text("?", font_size=48, color=HIGHLIGHT_COLOR)
        # Layout Fix (Question Mark): Snap to grid coordinate D4
        self.place_at_grid(question_mark, 'D4', scale_factor=0.8)
        
        self.play(
            self.lecture[2].animate.set_color(HIGHLIGHT_COLOR),
            Create(axis),
            *[squirrels[i].animate.move_to(target_positions[i]) for i in range(15)],
            run_time=2
        )
        
        self.play(
            target_squirrel.animate.set_color(HIGHLIGHT_COLOR),
            Write(question_mark),
            run_time=1
        )
        self.wait(3)
