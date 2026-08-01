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
        # Setup title and lecture lines
        title_text = "The Geometric Update: Slicing the Universe"
        lecture_lines = [
            "Suddenly, the robot detective sees a bright spark.",
            "Non-sparking possibilities are now impossible in our world.",
            "We discard the bottom sections of our probability square.",
            "Our universe shrinks to only the shaded regions.",
            "Only two possible paths to a spark remain."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors from previous sections
        RED_SPARK = "#FF5555"
        GREEN_SPARK = "#55FF55"
        NEAR_BLACK = "#111111"
        HIGHLIGHT_COLOR = "#FFFF00"
        
        # --- Create Geometric Components (Initial State) ---
        # Base dimensions (3x scaling to fit the right-side grid area)
        # Probability model: 20% Glitch-Bots (TP/NS-Red), 80% Normal-Bots (FP/NS-Green)
        # Spark Rates: Glitch 90%, Normal 10%
        tp_rect = Rectangle(width=0.6, height=2.7, fill_color=RED_SPARK, fill_opacity=1, stroke_width=1, stroke_color=WHITE)
        fp_rect = Rectangle(width=2.4, height=0.3, fill_color=GREEN_SPARK, fill_opacity=1, stroke_width=1, stroke_color=WHITE)
        ns_red_rect = Rectangle(width=0.6, height=0.3, fill_color=NEAR_BLACK, fill_opacity=1, stroke_width=1, stroke_color=WHITE)
        ns_green_rect = Rectangle(width=2.4, height=2.7, fill_color=NEAR_BLACK, fill_opacity=1, stroke_width=1, stroke_color=WHITE)
        
        # Assemble components into a 3.0x3.0 unit square
        tp_rect.move_to([0, 0, 0])
        fp_rect.next_to(tp_rect, RIGHT, buff=0).align_to(tp_rect, UP)
        ns_red_rect.next_to(tp_rect, DOWN, buff=0)
        ns_green_rect.next_to(fp_rect, DOWN, buff=0)
        
        universe_frame = Rectangle(width=3.0, height=3.0, stroke_color=WHITE, stroke_width=2)
        universe_frame.move_to(VGroup(tp_rect, fp_rect, ns_red_rect, ns_green_rect).get_center())
        
        probability_group = VGroup(tp_rect, fp_rect, ns_red_rect, ns_green_rect, universe_frame)
        
        # Resolve Issue 34: Position probability square in B2-D5 area
        self.place_in_area(probability_group, 'B2', 'D5', scale_factor=1.0)
        
        # Internal labels for sparking regions (TP and FP)
        tp_label = Text("TP", font_size=18, color=WHITE).move_to(tp_rect.get_center())
        fp_label = Text("FP", font_size=18, color=WHITE).move_to(fp_rect.get_center())
        
        # Functional groups for animation
        non_sparking_regions = VGroup(ns_red_rect, ns_green_rect)
        tp_group = VGroup(tp_rect, tp_label)
        fp_group = VGroup(fp_rect, fp_label)
        target_group = VGroup(tp_group, fp_group) # The "sparking" universe
        
        # Resolve Issue 33: Spark star at A4 with appropriate scale
        spark_star = Star(n=8, color=HIGHLIGHT_COLOR, fill_opacity=1)
        self.place_at_grid(spark_star, "A4", scale_factor=0.6)
        
        # Initial State Display
        self.add(probability_group, tp_label, fp_label)

        # === Animation for Lecture Line 1 ===
        # Suddenly, the robot detective sees a bright spark.
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        self.play(FadeIn(spark_star), Indicate(spark_star, color=WHITE, scale_factor=1.5))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Non-sparking possibilities are now impossible in our world.
        self.lecture[1].set_color(HIGHLIGHT_COLOR)
        self.play(
            non_sparking_regions.animate.set_fill(BLACK, opacity=1).set_stroke(opacity=0),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # We discard the bottom sections of our probability square.
        self.lecture[2].set_color(HIGHLIGHT_COLOR)
        self.play(
            FadeOut(non_sparking_regions),
            FadeOut(universe_frame),
            FadeOut(spark_star),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Our universe shrinks to only the shaded regions.
        self.lecture[3].set_color(HIGHLIGHT_COLOR)
        
        # Geometric update: slide remaining blocks to a common baseline
        # and center them in the new available area (C2-D5)
        tl_c2 = self.grid['C2']
        br_d5 = self.grid['D5']
        center_shrunk = (tl_c2 + br_d5) / 2
        
        self.play(
            fp_group.animate.align_to(tp_group, DOWN),
            run_time=1
        )
        self.play(
            target_group.animate.move_to(center_shrunk),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Only two possible paths to a spark remain.
        self.lecture[4].set_color(HIGHLIGHT_COLOR)
        
        # Brace and label indicating total evidence P(E)
        brace = Brace(target_group, DOWN, color=WHITE)
        brace_label = Text("Total Evidence: P(E)", font_size=22, color=WHITE)
        evidence_label_group = VGroup(brace, brace_label)
        brace_label.next_to(brace, DOWN, buff=0.1)
        
        # Resolve Issue 35: Positioning evidence label/brace in Row E
        self.place_in_area(evidence_label_group, 'E2', 'E5', scale_factor=0.8)
        
        self.play(
            Create(brace),
            Write(brace_label),
            run_time=1.5
        )
        self.wait(2)
