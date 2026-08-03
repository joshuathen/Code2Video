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

class Section6Scene(TeachingScene):
    def construct(self):
        # Title and Lecture Lines
        title = "Summary: The World's Most Inefficient Calculator"
        lines = [
            "Geometry links billiard ball physics to circular constants.",
            "It is a beautiful but inefficient way to calculate pi.",
            "One hundred billion kilograms yields only six digits."
        ]
        self.setup_layout(title, lines)

        # Colors
        BLOCK_A_COLOR = "#FF6347"  # Tomato
        BLOCK_B_COLOR = "#1E90FF"  # DodgerBlue
        CIRCLE_COLOR = "#00FA9A"   # MediumSpringGreen
        PI_COLOR = "#FFD700"       # Gold
        
        # Assets
        WALL_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg"
        BLOCK_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg"
        SCOREBOARD_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/scoreboard.svg"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Representing the connection: Circle and Blocks
        circle = Circle(radius=0.8, color=CIRCLE_COLOR)
        self.place_in_area(circle, "B2", "C3")
        
        block_a_icon = SVGMobject(BLOCK_ASSET, color=BLOCK_A_COLOR, fill_opacity=0.8)
        block_b_icon = SVGMobject(BLOCK_ASSET, color=BLOCK_B_COLOR, fill_opacity=0.8)
        blocks_vgroup = VGroup(block_a_icon, block_b_icon).arrange(RIGHT, buff=0.2)
        self.place_in_area(blocks_vgroup, "B4", "C5", scale_factor=0.6)
        
        link_arrow = Arrow(circle.get_right(), blocks_vgroup.get_left(), color=WHITE)
        
        self.play(
            Create(circle),
            DrawBorderThenFill(blocks_vgroup),
            Write(link_arrow),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Transition to "Massive Block B" scene
        self.play(FadeOut(circle, blocks_vgroup, link_arrow))
        
        # Visual setup: Wall and Floor
        wall = SVGMobject(WALL_ASSET, height=3, color=GRAY)
        self.place_at_grid(wall, "E1")
        
        floor = Line(self.grid["F1"], self.grid["F6"], color=GRAY)
        
        # Massive Block B and Small Block A
        small_block = SVGMobject(BLOCK_ASSET, height=0.4, color=BLOCK_A_COLOR, fill_opacity=0.8)
        self.place_at_grid(small_block, "E2")
        
        massive_block = SVGMobject(BLOCK_ASSET, height=1.5, color=BLOCK_B_COLOR, fill_opacity=0.8)
        self.place_in_area(massive_block, "D4", "F6") # Very large block
        
        self.play(
            FadeIn(wall),
            Create(floor),
            FadeIn(small_block),
            FadeIn(massive_block),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Scoreboard Asset
        scoreboard_bg = SVGMobject(SCOREBOARD_ASSET, height=1.0, color=WHITE)
        self.place_in_area(scoreboard_bg, "A3", "A6")
        
        # Score value tracker and number
        score_val = ValueTracker(0.0)
        # Using DecimalNumber for pi digits display
        scoreboard_text = DecimalNumber(
            score_val.get_value(),
            num_decimal_places=8,
            color=PI_COLOR
        ).scale(0.8)
        scoreboard_text.move_to(scoreboard_bg.get_center())
        
        # Label for scoreboard - Fixing Issue 38
        pi_label = Text("Collisions:", font_size=20, color=WHITE)
        self.place_in_area(pi_label, "A1", "A2", scale_factor=0.5)
        
        self.add(pi_label, scoreboard_bg, scoreboard_text)
        
        # Update function for scoreboard to simulate rapid counting
        scoreboard_text.add_updater(lambda m: m.set_value(score_val.get_value()))
        
        # Rapid fire collisions simulation
        self.play(
            score_val.animate.set_value(3.14159265),
            massive_block.animate.shift(LEFT * 1.5),
            small_block.animate.set_opacity(0.3).set_opacity(0.8).set_opacity(0.3).set_opacity(0.8),
            run_time=4,
            rate_func=linear
        )
        
        scoreboard_text.clear_updaters()
        
        # Final formatting of pi digits
        final_pi = Text("3.14159265...", font_size=36, color=PI_COLOR)
        self.place_in_area(final_pi, "A3", "A6")
        
        self.play(
            FadeOut(scoreboard_bg, scoreboard_text),
            FadeIn(final_pi)
        )
        self.wait(1)
        
        # Final fade out, leaving only Pi
        self.play(
            FadeOut(wall),
            FadeOut(floor),
            FadeOut(small_block),
            FadeOut(massive_block),
            FadeOut(pi_label),
            final_pi.animate.scale(1.5).move_to(self.grid["C3"]),
            run_time=2
        )
        
        self.wait(3)
