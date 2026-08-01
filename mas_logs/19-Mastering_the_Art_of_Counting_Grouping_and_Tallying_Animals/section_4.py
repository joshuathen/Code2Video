from manim import *
import numpy as np

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

class Section4Scene(TeachingScene):
    def construct(self):
        # Setup the layout with the given title and lecture lines
        lecture_lines = [
            "Let's turn our marks into colored blocks.",
            "Stack three yellow blocks for the lions.",
            "Five black blocks show our five penguins.",
            "Two grey blocks stand for the elephants.",
            "Now we have a clear animal chart."
        ]
        self.setup_layout("Visualizing Data: The Animal Pictograph", lecture_lines)

        # Asset paths
        LION_ICON_PATH = "/mmfs1/data/home/jthen/Code2Video/assets/icon/lion.svg"
        PENGUIN_ICON_PATH = "/mmfs1/data/home/jthen/Code2Video/assets/icon/penguin.svg"
        ELEPHANT_ICON_PATH = "/mmfs1/data/home/jthen/Code2Video/assets/icon/elephant.svg"

        # Colors
        COLOR_LION = "#FFFF00"
        COLOR_PENGUIN = "#404040"
        COLOR_ELEPHANT = "#808080"

        # Helper to create tally marks
        def create_tally(count, color):
            tally_group = VGroup()
            for i in range(count):
                if (i + 1) % 5 == 0:
                    # Diagonal slash for the 5th mark
                    slash = Line(start=[-0.2, -0.3, 0], end=[0.2, 0.3, 0], color=color, stroke_width=4)
                    tally_group.add(slash)
                else:
                    # Vertical mark
                    line = Line(start=[0, -0.3, 0], end=[0, 0.3, 0], color=color, stroke_width=4)
                    # Position vertical lines horizontally within the tally group
                    line.shift(RIGHT * (0.15 * (i % 5)))
                    tally_group.add(line)
            return tally_group

        # Helper to create a block (square)
        def create_block(color):
            return Square(side_length=0.7, fill_opacity=1, fill_color=color, stroke_color=WHITE, stroke_width=1)

        # === Animation for Lecture Line 1 ===
        # Let's turn our marks into colored blocks.
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # Create initial tally marks
        tally_lion = create_tally(3, COLOR_LION)
        tally_penguin = create_tally(5, COLOR_PENGUIN)
        tally_elephant = create_tally(2, COLOR_ELEPHANT)

        self.place_at_grid(tally_lion, "C2", scale_factor=1.0)
        self.place_at_grid(tally_penguin, "C3", scale_factor=1.0)
        self.place_at_grid(tally_elephant, "C4", scale_factor=1.0)

        self.play(Create(tally_lion), Create(tally_penguin), Create(tally_elephant))
        self.wait(1)

        # Create the blocks for morphing
        lion_blocks = VGroup(*[create_block(COLOR_LION) for _ in range(3)])
        penguin_blocks = VGroup(*[create_block(COLOR_PENGUIN) for _ in range(5)])
        elephant_blocks = VGroup(*[create_block(COLOR_ELEPHANT) for _ in range(2)])

        # Place them at the same positions as tallies for morphing
        for block in lion_blocks: block.move_to(tally_lion.get_center())
        for block in penguin_blocks: block.move_to(tally_penguin.get_center())
        for block in elephant_blocks: block.move_to(tally_elephant.get_center())

        self.play(
            ReplacementTransform(tally_lion, lion_blocks),
            ReplacementTransform(tally_penguin, penguin_blocks),
            ReplacementTransform(tally_elephant, elephant_blocks)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Stack three yellow blocks for the lions.
        self.play(self.lecture[1].animate.set_color(COLOR_LION))
        
        # Load lion icon
        lion_icon = SVGMobject(LION_ICON_PATH).set_color(COLOR_LION)
        self.place_at_grid(lion_icon, "F2", scale_factor=0.6)
        lion_label = Text("Lions", font_size=16, color=COLOR_LION).next_to(lion_icon, DOWN, buff=0.1)

        # Animate stacking
        stack_anims = []
        target_positions = ["E2", "D2", "C2"]
        for i, block in enumerate(lion_blocks):
            stack_anims.append(block.animate.move_to(self.grid[target_positions[i]]))

        self.play(FadeIn(lion_icon), FadeIn(lion_label), *stack_anims)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Five black blocks show our five penguins.
        # (Using Dark Grey #404040 as specified)
        self.play(self.lecture[2].animate.set_color(COLOR_PENGUIN))
        
        penguin_icon = SVGMobject(PENGUIN_ICON_PATH).set_color(COLOR_PENGUIN)
        self.place_at_grid(penguin_icon, "F3", scale_factor=0.6)
        penguin_label = Text("Penguins", font_size=16, color=COLOR_PENGUIN).next_to(penguin_icon, DOWN, buff=0.1)

        stack_anims_p = []
        target_positions_p = ["E3", "D3", "C3", "B3", "A3"]
        for i, block in enumerate(penguin_blocks):
            stack_anims_p.append(block.animate.move_to(self.grid[target_positions_p[i]]))

        self.play(FadeIn(penguin_icon), FadeIn(penguin_label), *stack_anims_p)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Two grey blocks stand for the elephants.
        self.play(self.lecture[3].animate.set_color(COLOR_ELEPHANT))

        elephant_icon = SVGMobject(ELEPHANT_ICON_PATH).set_color(COLOR_ELEPHANT)
        self.place_at_grid(elephant_icon, "F4", scale_factor=0.6)
        elephant_label = Text("Elephants", font_size=16, color=COLOR_ELEPHANT).next_to(elephant_icon, DOWN, buff=0.1)

        stack_anims_e = []
        target_positions_e = ["E4", "D4"]
        for i, block in enumerate(elephant_blocks):
            stack_anims_e.append(block.animate.move_to(self.grid[target_positions_e[i]]))

        self.play(FadeIn(elephant_icon), FadeIn(elephant_label), *stack_anims_e)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Now we have a clear animal chart.
        self.play(self.lecture[4].animate.set_color(WHITE))
        
        # Group components for a slight highlight/pulse effect
        chart = VGroup(
            lion_blocks, penguin_blocks, elephant_blocks,
            lion_icon, penguin_icon, elephant_icon,
            lion_label, penguin_label, elephant_label
        )
        
        self.play(chart.animate.scale(1.05), run_time=0.5)
        self.play(chart.animate.scale(1/1.05), run_time=0.5)
        self.wait(2)
