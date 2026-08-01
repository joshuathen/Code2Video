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

class Section7Scene(TeachingScene):
    def construct(self):
        # Define lecture lines and title
        title = "Summary and Intuition Check"
        lines = [
            "Bayes' Theorem turns evidence into updated knowledge.",
            "We filter the world through what we observe.",
            "Now you can update your mind with geometric precision."
        ]
        
        self.setup_layout(title, lines)
        
        # Colors
        COLOR_METER = "#32CD32"
        COLOR_CALC = "#FFFFFF"
        COLOR_TAKEAWAY = "#FFD700"

        # === Animation for Lecture Line 1 ===
        # Cloud-Bot meter jump 20% to 69%
        self.lecture[0].set_color(COLOR_METER)
        
        # Asset: Robot [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg]
        # Using SVGMobject for the robot icon
        try:
            robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        except:
            # Fallback to a simple circle if asset loading fails
            robot = Circle(color=WHITE).scale(0.4)
            robot.add(Dot().shift(LEFT*0.1+UP*0.1), Dot().shift(RIGHT*0.1+UP*0.1)) # Eyes
            
        self.place_at_grid(robot, "B1", scale_factor=0.6)
        
        # Meter Mobjects
        meter_width = 3.0
        meter_height = 0.6
        
        meter_bg = Rectangle(width=meter_width, height=meter_height, color="#FFFFFF", stroke_width=2)
        meter_fill = Rectangle(
            width=meter_width * 0.20, 
            height=meter_height, 
            fill_opacity=0.8, 
            fill_color=COLOR_METER, 
            stroke_width=0
        )
        meter_fill.align_to(meter_bg, LEFT)
        
        meter_group = VGroup(meter_bg, meter_fill)
        # Issue 42: Scale factor set to 0.7 for meter_group
        self.place_in_area(meter_group, "B2", "C5", scale_factor=0.7)
        
        belief_label = Text("Rain Probability", font_size=20, color="#FFFFFF")
        belief_label.next_to(meter_group, UP, buff=0.2)
        
        percent_val = ValueTracker(20)
        percent_text = DecimalNumber(20, unit="%", num_decimal_places=0, color="#FFFFFF")
        percent_text.scale(0.7)
        percent_text.next_to(meter_group, RIGHT, buff=0.2)
        
        # Meter Updater - L025: stretch_to_fit_width
        def update_meter(m):
            # Target width based on background's width to account for group scaling
            target_width = meter_bg.get_width() * (percent_val.get_value() / 100)
            if target_width < 0.01: target_width = 0.01
            m.stretch_to_fit_width(target_width, about_edge=LEFT)
            
        meter_fill.add_updater(update_meter)
        percent_text.add_updater(lambda d: d.set_value(percent_val.get_value()))
        
        self.play(FadeIn(robot), FadeIn(meter_group), FadeIn(belief_label), FadeIn(percent_text))
        self.wait(1) 
        
        # Update belief
        # L024: Use rate_functions prefix
        self.play(
            percent_val.animate.set_value(69),
            run_time=2,
            rate_func=rate_functions.ease_in_out_quad
        )
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Calculation: '0.18 divided by 0.26 equals 0.69'
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_CALC)
        
        # Issue 43: Fix calc_tex positioning to area D2-D5 and scale 1.0
        calc_tex = MathTex(
            r"\frac{0.18}{0.26} \approx 0.69",
            color=COLOR_CALC
        )
        self.place_in_area(calc_tex, "D2", "D5", scale_factor=1.0)
        
        self.wait(1)
        self.play(Write(calc_tex))
        self.play(Indicate(calc_tex, color=COLOR_CALC)) # L004: Indicate
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Highlight takeaway: 'Evidence updates belief through geometry'
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_TAKEAWAY)
        
        # Issue 44: Fix takeaway scale factor to 0.8
        takeaway = Text("Evidence updates belief\nthrough geometry", font_size=24, color=COLOR_TAKEAWAY, line_spacing=0.8)
        self.place_in_area(takeaway, "E2", "F5", scale_factor=0.8)
        
        self.wait(1)
        self.play(FadeIn(takeaway, shift=UP*0.3))
        self.play(Indicate(takeaway, color=COLOR_TAKEAWAY))
        
        # Final highlight
        self.play(
            Circumscribe(meter_group, color=COLOR_METER),
            Circumscribe(calc_tex, color=COLOR_CALC),
            Circumscribe(takeaway, color=COLOR_TAKEAWAY),
            run_time=2
        )
        
        self.wait(3)
