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

class Section2Scene(TeachingScene):
    def construct(self):
        # 1. Setup Layout
        lines = [
            "The variable 's' controls how fast terms shrink.",
            "Larger 's' values force the sum to converge.",
            "We call this 's' the exponent of decay."
        ]
        self.setup_layout("Prerequisite: The Power of 's'", lines)

        # 2. Objects Creation
        # Formula: zeta(s) = sum 1/n^s in white #FFFFFF
        zeta_formula = MarkupText(
            '<i>ζ</i>(s) = <span size="x-large">∑</span> 1/n<sup>s</sup>',
            color="#FFFFFF"
        )
        self.place_in_area(zeta_formula, 'A1', 'C6', scale_factor=0.8)

        # ValueTracker for 's'
        s_tracker = ValueTracker(1.0)

        # Slider indicator: 's =' label and value
        s_label = Text("s =", color="#FFFFFF")
        s_value = DecimalNumber(1.0, color="#FFFFFF", num_decimal_places=1, mob_class=Text)
        s_value.add_updater(lambda d: d.set_value(s_tracker.get_value()))
        
        s_group = VGroup(s_label, s_value).arrange(RIGHT, buff=0.2)
        self.place_at_grid(s_group, 'D3', scale_factor=0.8)

        # Plot area
        axes = Axes(
            x_range=[1, 5, 1],
            y_range=[0, 1.5, 0.5],
            axis_config={"color": GREY_C},
            tips=False
        )
        self.place_in_area(axes, 'E1', 'F6', scale_factor=0.6)

        # Curve: y = 1/x^s in cyan #00FFFF
        curve = always_redraw(lambda: axes.plot(
            lambda x: 1 / (x**s_tracker.get_value()),
            x_range=[1, 5],
            color="#00FFFF"
        ))

        # Dynamic Area
        def get_area_color():
            val = s_tracker.get_value()
            if val >= 1.5:
                return "#00FF00"  # Solid Green
            elif val < 1.0:
                # Flashing Red
                if int(self.time * 6) % 2 == 0:
                    return "#FF0000"
                else:
                    return "#660000"
            else:
                return "#00FFFF" # Neutral Cyan

        area = always_redraw(lambda: axes.get_area(
            curve,
            x_range=[1, 5],
            color=get_area_color(),
            opacity=0.5
        ))

        # === Animation for Lecture Line 1 ===
        # Line color Cyan
        self.lecture[0].set_color("#00FFFF")
        self.add(zeta_formula, axes, s_group)
        self.play(Create(curve), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line color Green
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#00FF00")
        self.add(area)
        # Move slider to 2.0; curve flattens; area fills with solid green
        self.play(s_tracker.animate.set_value(2.0), run_time=3)
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Line color Red
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FF0000")
        # Move slider to 0.8; area under curve fills with flashing red
        self.play(s_tracker.animate.set_value(0.8), run_time=3)
        self.wait(3)
