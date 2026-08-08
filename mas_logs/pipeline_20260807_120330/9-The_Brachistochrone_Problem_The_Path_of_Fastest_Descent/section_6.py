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
        # Data from storyboard
        title = "The Tautochrone Property & Real-world Application"
        lines = [
            "The cycloid is also a tautochrone curve.",
            "Objects released from any height finish together.",
            "This property is used in clocks and engineering."
        ]
        self.setup_layout(title, lines)

        # Colors
        GREEN_RAMP = "#00FF00"
        RED_MARBLE = "#FF0000"
        BLUE_MARBLE = "#0000FF"
        WHITE_MARBLE = "#FFFFFF"
        FLASH_COLOR = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        # "The cycloid is also a tautochrone curve."
        self.play(self.lecture[0].animate.set_color(GREEN_RAMP))
        
        # Cycloid ramp parameters
        # Parametric equations: x = r(t - sin t), y = -r(1 - cos t)
        r = 0.55
        def ramp_func(t):
            # t from 0 to PI
            return np.array([r * (t - np.sin(t)), -r * (1 - np.cos(t)), 0])

        ramp = ParametricFunction(ramp_func, t_range=[0, PI], color=GREEN_RAMP)
        # Shift and place - Applied Fix for Issue 33: Move to Column 2-6 to avoid crowding
        self.place_in_area(ramp, "B2", "E6", scale_factor=0.8)
        
        self.play(Create(ramp))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Objects released from any height finish together."
        self.play(self.lecture[1].animate.set_color(FLASH_COLOR))

        # Marbles at different heights (proportions along the curve)
        props = [0.15, 0.45, 0.75]
        marbles = VGroup(
            Dot(color=RED_MARBLE, radius=0.12),
            Dot(color=BLUE_MARBLE, radius=0.12),
            Dot(color=WHITE_MARBLE, radius=0.12)
        )
        
        # Initial positions
        for i, m in enumerate(marbles):
            m.move_to(ramp.point_from_proportion(props[i]))
            
        self.play(FadeIn(marbles))
        self.wait(0.5)

        # Slide animation: all marbles reach the end (proportion 1.0) simultaneously
        def slide_animation(mobject, start_prop):
            return UpdateFromAlphaFunc(
                mobject,
                lambda m, alpha: m.move_to(
                    ramp.point_from_proportion(interpolate(start_prop, 1.0, alpha))
                )
            )

        self.play(
            slide_animation(marbles[0], props[0]),
            slide_animation(marbles[1], props[1]),
            slide_animation(marbles[2], props[2]),
            run_time=2.5,
            rate_func=bezier([0.4, 0, 1, 1])
        )

        # Simultaneous collision flash at the bottom
        flash = Flash(ramp.get_end(), color=FLASH_COLOR, line_length=0.4, num_lines=12, flash_radius=0.3)
        self.play(flash)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "This property is used in clocks and engineering."
        self.play(self.lecture[2].animate.set_color(BLUE_MARBLE))
        
        # Subtle emphasis: have marbles pulse or settle
        self.play(
            *[m.animate.scale(1.2).set_opacity(0.8) for m in marbles],
            run_time=0.5
        )
        self.play(
            *[m.animate.scale(1/1.2).set_opacity(0.6) for m in marbles],
            run_time=0.5
        )
        
        self.wait(2)
