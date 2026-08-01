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
        # Setup the layout with the specific section title and lecture lines
        title = "The Grand Connection: The Fundamental Theorem"
        lecture_lines = [
            "Differentiation breaks a whole into its changing parts.",
            "Integration sums these tiny parts back into a whole.",
            "These two processes are actually opposites of each other.",
            "Like addition and subtraction, they undo one another perfectly.",
            "This powerful link is the Fundamental Theorem of Calculus."
        ]
        self.setup_layout(title, lecture_lines)

        # Colors as defined in the storyboard/animations
        COLOR_CASTLE = "#FFFFFF"
        COLOR_BLOCKS = "#FF0000"
        COLOR_INVERSE = "#00FF00"
        COLOR_THEOREM = "#FFFF00"

        # Asset Paths
        CASTLE_PATH = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/castle.svg"
        BLOCKS_PATH = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg"

        # Initialize labels
        label_diff = Text("Differentiation", font_size=24, color=COLOR_BLOCKS)
        label_int = Text("Integration", font_size=24, color=COLOR_CASTLE)
        label_inverse = Text("Inverse Operations", font_size=24, color=COLOR_INVERSE)
        theorem_title = Text("Fundamental Theorem of Calculus", font_size=32, color=COLOR_THEOREM)

        # === Animation for Lecture Line 1 ===
        # Differentiation breaks a whole into its changing parts.
        self.play(self.lecture[0].animate.set_color(COLOR_BLOCKS))
        
        # Issue 40: Use place_in_area for castle_blocks in B3-D5
        # Issue 27: Load castle asset
        castle_blocks = SVGMobject(CASTLE_PATH).set_color(COLOR_CASTLE)
        self.place_in_area(castle_blocks, 'B3', 'D5', scale_factor=0.8)
        self.play(FadeIn(castle_blocks))
        
        # Issue 41: Position label_diff at E3
        self.place_at_grid(label_diff, 'E3', scale_factor=1.0)
        self.play(Write(label_diff))
        
        # Issue 27: Transform castle to broken blocks asset
        broken_blocks = SVGMobject(BLOCKS_PATH).set_color(COLOR_BLOCKS)
        broken_blocks.move_to(castle_blocks.get_center()).scale_to_fit_width(castle_blocks.width)
        
        self.play(ReplacementTransform(castle_blocks, broken_blocks))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Integration sums these tiny parts back into a whole.
        self.play(self.lecture[1].animate.set_color(COLOR_CASTLE))
        self.play(FadeOut(label_diff))
        
        # Issue 41: Position label_int at E3
        self.place_at_grid(label_int, 'E3', scale_factor=1.0)
        self.play(Write(label_int))
        
        # Issue 40: Reassemble into castle in area B3-D5
        reassembled_castle = SVGMobject(CASTLE_PATH).set_color(COLOR_CASTLE)
        self.place_in_area(reassembled_castle, 'B3', 'D5', scale_factor=0.8)
        
        self.play(ReplacementTransform(broken_blocks, reassembled_castle))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # These two processes are actually opposites of each other.
        self.play(self.lecture[2].animate.set_color(COLOR_INVERSE))
        self.play(FadeOut(label_int))
        
        # Issue 42: label_inverse area E4-E6
        self.place_in_area(label_inverse, 'E4', 'E6', scale_factor=1.0)
        self.play(Write(label_inverse))
        
        # Double headed arrow indicating relationship
        arrow = DoubleArrow(self.grid["C3"], self.grid["C5"], color=COLOR_INVERSE, stroke_width=5)
        self.play(GrowFromCenter(arrow))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Like addition and subtraction, they undo one another perfectly.
        self.play(self.lecture[3].animate.set_color(COLOR_INVERSE))
        
        # Loop animation to show undoing/reversibility using transforms
        undo_blocks = SVGMobject(BLOCKS_PATH).set_color(COLOR_BLOCKS)
        undo_blocks.move_to(reassembled_castle.get_center()).scale_to_fit_width(reassembled_castle.width)
        
        self.play(
            ReplacementTransform(reassembled_castle, undo_blocks),
            Indicate(arrow, color=COLOR_INVERSE),
            run_time=1
        )
        
        final_castle = SVGMobject(CASTLE_PATH).set_color(COLOR_CASTLE)
        final_castle.move_to(undo_blocks.get_center()).scale_to_fit_width(undo_blocks.width)
        
        self.play(
            ReplacementTransform(undo_blocks, final_castle),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # This powerful link is the Fundamental Theorem of Calculus.
        self.play(self.lecture[4].animate.set_color(COLOR_THEOREM))
        
        # Issue 42: theorem_title area A2-A6
        self.place_in_area(theorem_title, 'A2', 'A6', scale_factor=1.0)
        self.play(FadeIn(theorem_title, shift=UP))
        self.play(Indicate(theorem_title, color=COLOR_THEOREM, scale_factor=1.1))
        self.wait(2)
