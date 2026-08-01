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
        # Data
        title = "Prerequisite: Weights and Predictions"
        lines = [
            "Pixel uses knobs called Weights to make predictions.",
            "Adjusting these knobs changes how he sees data.",
            "Learning means finding the perfect knob settings."
        ]
        
        self.setup_layout(title, lines)
        
        # Colors
        C_L1 = YELLOW_A
        C_L2 = GREEN_A
        C_L3 = BLUE_A
        
        PIXEL_COLOR = BLUE_D
        KNOB_COLOR = GRAY_A

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(C_L1))
        
        # Create Pixel (Robot Character)
        pixel_body = RoundedRectangle(width=1.5, height=2.0, corner_radius=0.2, color=PIXEL_COLOR, fill_opacity=0.5)
        pixel_head = Square(side_length=0.8, color=PIXEL_COLOR, fill_opacity=0.8)
        pixel_head.next_to(pixel_body, UP, buff=0.1)
        pixel_eye_l = Dot(radius=0.1, color=WHITE).move_to(pixel_head.get_center() + LEFT*0.2 + UP*0.1)
        pixel_eye_r = Dot(radius=0.1, color=WHITE).move_to(pixel_head.get_center() + RIGHT*0.2 + UP*0.1)
        pixel = VGroup(pixel_body, pixel_head, pixel_eye_l, pixel_eye_r)
        
        # Issue 31: Place Pixel in grid area B2 to D3
        self.place_in_area(pixel, "B2", "D3", scale_factor=1.0)
        
        # Issue 25: Load Knob Asset
        # [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/kn.svg]
        knob_asset = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/kn.svg", color=KNOB_COLOR).scale(0.35)
        knob_label = Text("Weight", font_size=16, color=C_L1).next_to(knob_asset, DOWN, buff=0.1)
        knob = VGroup(knob_asset, knob_label)
        
        # Issue 32: Place knob at C2
        self.place_at_grid(knob, "C2", scale_factor=1.0)
        
        self.play(FadeIn(pixel), FadeIn(knob))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(C_L2)
        )
        
        # Guess Screen
        screen_bg = Rectangle(width=2.5, height=1.8, color=GREY_E, fill_opacity=0.9)
        screen_bg.set_stroke(C_L2, 2)
        screen_title = Text("Guess (kg)", font_size=20, color=WHITE)
        
        # Value tracker for interactive guess number
        weight_tracker = ValueTracker(1.0)
        guess_num = DecimalNumber(1.0, num_decimal_places=1, color=C_L2, font_size=36, mob_class=Text)
        guess_num.add_updater(lambda d: d.set_value(weight_tracker.get_value()))
        
        screen_content = VGroup(screen_title, guess_num).arrange(DOWN, buff=0.3)
        screen = VGroup(screen_bg, screen_content)
        
        # Issue 33: Position Screen in grid area B5 to D6
        self.place_in_area(screen, "B5", "D6", scale_factor=1.0)
        screen_content.move_to(screen_bg.get_center())
        
        self.play(FadeIn(screen))
        
        # Asset rotates, and the 'Guess' number increases
        # We rotate the knob_asset (SVG) specifically
        self.play(
            Rotate(knob_asset, angle=-PI*0.8, about_point=knob_asset.get_center()),
            weight_tracker.animate.set_value(10.0),
            run_time=2.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(C_L3)
        )
        
        # Turning the knob back to "Learning" / "Finding perfect settings"
        self.play(
            Rotate(knob_asset, angle=PI*0.4, about_point=knob_asset.get_center()),
            weight_tracker.animate.set_value(5.0),
            run_time=2.5
        )
        self.wait(2)
