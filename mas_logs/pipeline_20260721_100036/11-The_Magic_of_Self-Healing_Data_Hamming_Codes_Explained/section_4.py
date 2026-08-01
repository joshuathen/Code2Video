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

class Section4Scene(TeachingScene):
    def construct(self):
        # Fetch lecture lines from storyboard
        lecture_lines = [
            "Each circle reports if its parity is correct.",
            "If one circle fails, the error is inside it.",
            "Overlapping failures pinpoint the exact corrupted bit.",
            "Binary coordinates act like a GPS for errors.",
            "We flip the broken bit back to heal it."
        ]
        
        self.setup_layout("The Interrogation: Locating the Error", lecture_lines)

        # Define Colors as per requirements
        COLOR_ERROR = "#FF0000"  # Red
        COLOR_OK = "#00FF00"     # Green
        COLOR_HIGHLIGHT = "#FFFF00"
        HEX_RED = "#FF0000"
        HEX_GREEN = "#00FF00"
        HEX_BLUE = "#5555FF"

        # === Animation for Lecture Line 1 ===
        # Each circle reports if its parity is correct.
        
        # Create Circles
        circle_red = Circle(radius=1.3, color=HEX_RED, stroke_width=4)
        circle_green = Circle(radius=1.3, color=HEX_GREEN, stroke_width=4)
        circle_blue = Circle(radius=1.3, color=HEX_BLUE, stroke_width=4)

        # Positioning based on grid (Moved down as per Issue 32 and scaled as per Issue 33)
        self.place_at_grid(circle_red, 'C3', scale_factor=1.3)
        self.place_at_grid(circle_green, 'C5', scale_factor=1.3)
        self.place_at_grid(circle_blue, 'E4', scale_factor=1.3)

        # Create Bits inside the Venn diagram
        # Adjusted positions relative to the shifted circles
        bit_data = [
            ("P1", "0", "C2"),
            ("P2", "1", "C6"),
            ("P4", "0", "F4"),
            ("D1", "1", "C4"), # THE ERROR BIT (Intersection of R and G, outside B)
            ("D2", "0", "D5"),
            ("D3", "1", "D3"),
            ("D4", "1", "D4"),
        ]
        
        bits_group = VGroup()
        bit_objects = {}
        for label, val, pos in bit_data:
            txt = Text(val, font_size=24, color=WHITE)
            self.place_at_grid(txt, pos)
            bits_group.add(txt)
            bit_objects[label] = txt

        self.add(circle_red, circle_green, circle_blue, bits_group)
        
        # Trigger line 1
        self.play(self.lecture[0].animate.set_color(COLOR_OK), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # If one circle fails, the error is inside it.
        self.play(self.lecture[1].animate.set_color(COLOR_ERROR), run_time=1)
        
        # Highlight Red failure
        pulse_r = circle_red.copy().set_stroke(width=10, color=COLOR_ERROR)
        self.play(
            pulse_r.animate.scale(1.1).set_opacity(0),
            rate_func=there_and_back,
            run_time=1.5
        )
        self.remove(pulse_r)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Overlapping failures pinpoint the exact corrupted bit.
        self.play(self.lecture[2].animate.set_color(COLOR_HIGHLIGHT), run_time=1)

        # Blue remains green (indicates correct parity)
        self.play(circle_blue.animate.set_color(COLOR_OK), run_time=0.5)

        # Pulsing both Red and Green to show dual failure
        pulse_r = circle_red.copy().set_stroke(width=12, color=COLOR_ERROR)
        pulse_g = circle_green.copy().set_stroke(width=12, color=COLOR_ERROR)
        
        self.play(
            AnimationGroup(
                pulse_r.animate.scale(1.2).set_opacity(0),
                pulse_g.animate.scale(1.2).set_opacity(0),
                lag_ratio=0.1
            ),
            rate_func=lambda t: np.sin(t * PI),
            run_time=2
        )
        self.remove(pulse_r, pulse_g)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Binary coordinates act like a GPS for errors.
        self.play(self.lecture[3].animate.set_color(COLOR_HIGHLIGHT), run_time=1)

        error_bit = bit_objects["D1"] # Located at C4
        
        # Pointer highlight at the intersection
        gps_circle = Circle(radius=0.5, color=COLOR_HIGHLIGHT).move_to(error_bit)
        self.play(Create(gps_circle))
        
        # Pulsing the bit in the intersection
        self.play(
            error_bit.animate.scale(1.8).set_color(COLOR_ERROR),
            run_time=0.6
        )
        self.play(
            error_bit.animate.scale(1/1.8),
            run_time=0.4
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # We flip the broken bit back to heal it.
        self.play(self.lecture[4].animate.set_color(COLOR_OK), run_time=1)

        # Flip "1" to "0"
        healed_val = Text("0", font_size=24, color=COLOR_OK)
        healed_val.move_to(error_bit)
        
        self.play(
            Transform(error_bit, healed_val),
            FadeOut(gps_circle),
            run_time=1
        )
        
        # Restore circles to original colors to show healing complete
        self.play(
            circle_red.animate.set_color(HEX_RED),
            circle_green.animate.set_color(HEX_GREEN),
            circle_blue.animate.set_color(HEX_BLUE),
            run_time=1
        )
        
        self.wait(3)
