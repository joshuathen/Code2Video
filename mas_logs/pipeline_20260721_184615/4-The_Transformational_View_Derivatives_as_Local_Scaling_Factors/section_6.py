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
        self.setup_layout(
            "Summary: Derivatives as the Local 'Speedometer'",
            [
                "Heat maps reveal where the function expands or contracts.",
                "Red indicates local stretching while blue shows space squishing.",
                "The derivative is the local scaling factor of space."
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Highlight lecture line 1
        self.lecture[0].set_color("#FFFF00") # Yellow highlight
        
        # Number line with a moving point and a heat map background
        number_line = NumberLine(
            x_range=[0, 2, 0.5],
            length=3,
            include_numbers=True,
            font_size=18,
            color="#FFFFFF",
            stroke_width=2
        )
        # Fix overlap by shifting to C4-C6 using grid
        self.place_in_area(number_line, "C4", "C6")
        
        # Moving point (Asset) on the number line
        speed_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speed.svg")
        speed_icon.set_color("#00FFFF")
        speed_icon.scale(0.3)
        dot_tracker = ValueTracker(0.2)
        speed_icon.add_updater(lambda d: d.move_to(number_line.n2p(dot_tracker.get_value())))
        
        # Initial Heat map (neutral)
        heat_map_neutral = Rectangle(
            width=3, height=0.4, 
            fill_opacity=0.3, 
            stroke_width=0, 
            fill_color="#888888"
        )
        heat_map_neutral.move_to(number_line.get_center())
        
        self.play(Create(number_line), FadeIn(heat_map_neutral))
        self.add(speed_icon)
        self.play(dot_tracker.animate.set_value(1.8), run_time=2, rate_func=linear)
        self.play(dot_tracker.animate.set_value(0.2), run_time=2, rate_func=linear)
        
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # Transition lecture highlight
        self.play(
            self.lecture[0].animate.set_color("#FFFFFF"),
            self.lecture[1].animate.set_color("#00FFFF") # Cyan highlight
        )
        
        # Change background to red where f'(x) > 1 and blue where f'(x) < 1
        blue_segment = Rectangle(
            width=1.5, height=0.4, 
            fill_opacity=0.6, 
            stroke_width=0, 
            fill_color="#0000FF"
        )
        
        red_segment = Rectangle(
            width=1.5, height=0.4, 
            fill_opacity=0.6, 
            stroke_width=0, 
            fill_color="#FF0000"
        )
        
        # Group them to align perfectly
        heat_map_colored = VGroup(blue_segment, red_segment).arrange(RIGHT, buff=0)
        heat_map_colored.move_to(number_line.get_center())

        # Labels for blue and red regions
        blue_label = Text("f' < 1", font_size=18, color="#0000FF")
        red_label = Text("f' > 1", font_size=18, color="#FF0000")
        
        # Fix misalignment and text obstruction
        self.place_at_grid(blue_label, "D4", scale_factor=0.7)
        self.place_at_grid(red_label, "D6", scale_factor=0.6)

        self.play(
            FadeOut(heat_map_neutral),
            FadeIn(heat_map_colored),
            FadeIn(blue_label),
            FadeIn(red_label)
        )
        
        # Let the point move through the regions
        self.play(dot_tracker.animate.set_value(1.8), run_time=3, rate_func=linear)
        self.wait(2.0)

        # === Animation for Lecture Line 3 ===
        # Transition lecture highlight
        self.play(
            self.lecture[1].animate.set_color("#FFFFFF"),
            self.lecture[2].animate.set_color("#FFFF00") # Yellow highlight
        )
        
        # Flash the text 'Derivative = Local Scaling'
        summary_text = Text("Derivative = Local Scaling", font_size=24, color="#FFFF00")
        self.place_in_area(summary_text, "B4", "B6")
        
        self.play(FadeIn(summary_text))
        self.play(Indicate(summary_text, color="#FFFF00", scale_factor=1.2))
        
        # Final slow movement
        self.play(dot_tracker.animate.set_value(0.5), run_time=2)
        
        self.wait(2.0)

        # Cleanup for end of video
        self.play(FadeOut(VGroup(number_line, speed_icon, heat_map_colored, blue_label, red_label, summary_text)))
        self.wait(1.0)
