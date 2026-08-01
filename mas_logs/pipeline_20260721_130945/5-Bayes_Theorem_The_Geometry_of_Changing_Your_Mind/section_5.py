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
        title = "The Posterior: Re-normalizing the World"
        lines = [
            "We must find the new probability of a Glitch-Bot.",
            "Compare the Glitch-Spark area to the total spark area.",
            "We stretch these remaining areas to fill the square.",
            "This re-normalization reveals our updated posterior belief.",
            "The Glitch-Bot probability is now roughly sixty-nine percent."
        ]
        self.setup_layout(title, lines)

        # Colors
        GLITCH_COLOR = "#3498DB" # Initial color for Glitch (Blue)
        NORMAL_COLOR = "#95A5A6" # Initial color for Normal (Gray)
        POSTERIOR_GOLD = "#F1C40F" # Final Gold color
        
        # Area Constants:
        # P(Glitch & Spark) = 0.18
        # P(Normal & Spark) = 0.08
        # Total Evidence = 0.26
        # Ratio: 0.18/0.26 = 0.692 (Glitch), 0.08/0.26 = 0.308 (Normal)
        total_width = 3.0
        glitch_width = 0.692 * total_width
        normal_width = 0.308 * total_width
        rect_height = 3.0

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Create initial rects representing the spark areas
        # These are "slices" of the original unit square. 
        # Height is initially small to represent sub-areas from previous sections.
        glitch_rect = Rectangle(width=glitch_width, height=1.0, fill_opacity=0.8, color=GLITCH_COLOR, fill_color=GLITCH_COLOR)
        normal_rect = Rectangle(width=normal_width, height=0.4, fill_opacity=0.8, color=NORMAL_COLOR, fill_color=NORMAL_COLOR)
        
        spark_group = VGroup(glitch_rect, normal_rect).arrange(RIGHT, buff=0, aligned_edge=DOWN)
        # Issue 37: Increase scale for visibility
        self.place_in_area(spark_group, "B2", "E5", scale_factor=1.1)
        
        self.play(FadeIn(spark_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # No specific visual change requested for "Compare", 
        # but we can highlight them slightly.
        self.play(Indicate(spark_group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Scale heights of both blocks to fill the vertical space (re-stretching)
        # We also scale the width to ensure they form a unit-like square structure.
        # Issue 38: Final square area scale 0.85 to avoid labels feeling cramped.
        
        # We'll do the stretch first, then the scale/re-position.
        self.play(
            glitch_rect.animate.stretch_to_fit_height(rect_height).align_to(spark_group, DOWN),
            normal_rect.animate.stretch_to_fit_height(rect_height).align_to(spark_group, DOWN),
            run_time=1.5
        )
        
        final_square_area = VGroup(glitch_rect, normal_rect)
        self.play(
            self.place_in_area(final_square_area, "B2", "E5", scale_factor=0.85).animate,
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Asset: Robot icon
        robot_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        robot_icon.set_color(WHITE)
        # Place robot in upper half of the glitch region
        robot_icon.scale(0.4)
        robot_icon.move_to(glitch_rect.get_center() + UP * 0.5)
        
        boundary_line = Line(
            glitch_rect.get_critical_point(UR),
            glitch_rect.get_critical_point(DR),
            color=WHITE,
            stroke_width=4
        )
        
        self.play(
            glitch_rect.animate.set_fill(POSTERIOR_GOLD).set_color(POSTERIOR_GOLD),
            Create(boundary_line),
            FadeIn(robot_icon),
            run_time=1.5
        )
        self.play(Indicate(boundary_line, color=WHITE))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Posterior percentage '69.2%'
        # L022: Use Text for stability.
        posterior_text = Text("69.2%", font_size=32, color=BLACK, weight=BOLD)
        # Place below robot icon
        posterior_text.move_to(glitch_rect.get_center() + DOWN * 0.5)
        
        # Label for normal side (30.8%)
        normal_label = Text("30.8%", font_size=24, color=WHITE).scale(0.8)
        normal_label.move_to(normal_rect.get_center())
        
        self.play(
            Write(posterior_text),
            FadeIn(normal_label),
            run_time=1.5
        )
        self.wait(1)
        
        self.play(Indicate(posterior_text, scale_factor=1.1, color=POSTERIOR_GOLD))
        self.wait(3)
