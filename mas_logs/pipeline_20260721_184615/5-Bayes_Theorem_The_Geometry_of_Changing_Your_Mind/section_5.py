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

class Section5Scene(TeachingScene):
    def construct(self):
        title_str = "Normalization: The New Universe"
        lines = [
            "These remaining slivers don't fill the square.",
            "But they now represent our entire reality.",
            "We must rescale them to sum to one.",
            "This vertical stretching creates our new probability space.",
            "The larger sliver becomes our updated posterior belief."
        ]
        self.setup_layout(title_str, lines)

        # Colors
        GREEN = "#00FF00"
        RED = "#FF0000"
        WHITE = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # "These remaining slivers don't fill the square."
        self.lecture[0].set_color(GREEN)
        
        # Create a unit square frame for reference
        frame = Rectangle(width=3, height=3, color=WHITE, stroke_width=2)
        self.place_in_area(frame, "B2", "E5")
        
        # The 'slivers' - partial height rectangles
        green_sliver = Rectangle(width=1.8, height=0.6, fill_opacity=0.8, fill_color=GREEN, stroke_width=1)
        red_sliver = Rectangle(width=1.2, height=0.6, fill_opacity=0.8, fill_color=RED, stroke_width=1)
        
        slivers = VGroup(green_sliver, red_sliver).arrange(RIGHT, buff=0)
        # Position them at the bottom of the frame
        slivers.move_to(frame.get_bottom(), aligned_edge=DOWN)
        
        self.play(Create(frame))
        self.play(FadeIn(slivers))
        self.wait(1.0)

        # === Animation for Lecture Line 2 ===
        # "But they now represent our entire reality."
        self.lecture[1].set_color(GREEN)
        
        reality_box = SurroundingRectangle(slivers, color=WHITE, buff=0.1)
        reality_label = Text("New Universe", font_size=18, color=WHITE)
        reality_label.next_to(reality_box, UP, buff=0.2)
        
        self.play(Create(reality_box), Write(reality_label))
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # "We must rescale them to sum to one."
        self.lecture[2].set_color(GREEN)
        
        # Highlight scaling requirement
        scaling_arrow = Arrow(start=slivers.get_top(), end=frame.get_top(), color=WHITE)
        self.play(GrowArrow(scaling_arrow))
        self.wait(1.0)
        self.play(FadeOut(scaling_arrow), FadeOut(reality_label), FadeOut(reality_box))
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        # "This vertical stretching creates our new probability space."
        self.lecture[3].set_color(GREEN)
        
        norm_text = Text("Normalization: Adjusting to the New Total", font_size=20, color=WHITE)
        # Fix [VideoCritic] Issue 36: Re-position norm_text
        self.place_in_area(norm_text, 'A1', 'A6', scale_factor=0.6)

        # [Asset Integration] Issue 25: based.svg
        # Loading the asset
        based_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/based.svg")
        self.place_at_grid(based_icon, "A6", scale_factor=0.3)
        # Shift it a bit to be next to the text or just leave it at A6 as text spans A1-A6.
        # Actually A1-A6 is the whole top row. Let's put text slightly left and icon right.
        norm_group = VGroup(norm_text, based_icon).arrange(RIGHT, buff=0.2)
        self.place_in_area(norm_group, 'A1', 'A6', scale_factor=0.6)

        # Vertical stretching to fill frame height (3.0)
        # Align correctly inside frame
        self.play(
            green_sliver.animate.stretch_to_fit_height(3.0).move_to(frame.get_left() + RIGHT * 0.9, aligned_edge=LEFT),
            red_sliver.animate.stretch_to_fit_height(3.0).move_to(frame.get_right() + LEFT * 0.6, aligned_edge=RIGHT),
            Write(norm_group),
            run_time=2
        )
        self.wait(1.5)

        # === Animation for Lecture Line 5 ===
        # "The larger sliver becomes our updated posterior belief."
        self.lecture[4].set_color(GREEN)
        
        # Highlight green sliver
        self.play(Indicate(green_sliver, color=GREEN))
        
        # Fix [VideoCritic] Issue 37: total_area_label positioning
        total_area_label = Text("Total Area = 1.0", font_size=24, color=WHITE)
        self.place_in_area(total_area_label, 'F2', 'F5', scale_factor=0.8)
        
        # Fix [VideoCritic] Issue 38: posterior_label positioning
        # Green sliver is on the left half of the frame (columns 2-3 roughly)
        posterior_label = Text("Posterior", font_size=22, color=GREEN)
        self.place_in_area(posterior_label, 'C2', 'D3', scale_factor=0.6)
        
        self.play(Write(total_area_label), Write(posterior_label))
        self.wait(3.0)
