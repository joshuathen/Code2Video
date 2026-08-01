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

class Section1Scene(TeachingScene):
    def construct(self):
        title_text = "The Problem: Digital Whispers and Noise"
        lecture_lines = [
            "Digital data travels across long distances through noisy channels.",
            "Interference like solar flares can flip a zero to one.",
            "Bitsy needs a way to fix errors without retransmission."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors based on storyboard
        COLOR_1 = "#FFFFFF"
        COLOR_2 = "#FF0000"
        COLOR_3 = "#FF0000"

        # === Animation for Lecture Line 1 ===
        # Display a sequence of bits '1 0 1 1' representing a message from Bitsy the Robot.
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg
        self.play(self.lecture[0].animate.set_color(COLOR_1))
        
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        self.place_at_grid(robot, "B2", scale_factor=0.6)
        
        bits_values = ["1", "0", "1", "1"]
        bits_mobjects = VGroup(*[Text(b, font_size=48, color=COLOR_1) for b in bits_values])
        
        # Placing bits in Row C (C2 to C5) to utilize lower grid area as per VideoCritic (Issue 29)
        for i, bit in enumerate(bits_mobjects):
            self.place_at_grid(bit, f"C{i+2}")
            
        self.play(FadeIn(robot), Write(bits_mobjects))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Introduce a jagged red 'noise' wave passing through the bits, causing the third bit to flicker.
        self.play(self.lecture[1].animate.set_color(COLOR_2))
        
        # Create a jagged noise wave passing through Row C
        noise_points = [
            self.grid["C1"] + LEFT * 0.5,
            self.grid["C2"] + UP * 0.4,
            self.grid["C3"] + DOWN * 0.4,
            self.grid["C4"] + UP * 0.6,
            self.grid["C5"] + DOWN * 0.5,
            self.grid["C6"] + RIGHT * 0.5
        ]
        noise_wave = VMobject(color=COLOR_2).set_points_as_corners(noise_points)
        
        self.play(Create(noise_wave), run_time=1.5)
        
        # Flicker the third bit (bits_mobjects[2]) at C4
        # L008: Use set_fill_opacity/set_stroke_opacity instead of .opacity
        for _ in range(3):
            self.play(bits_mobjects[2].animate.set_fill_opacity(0.3), run_time=0.1)
            self.play(bits_mobjects[2].animate.set_fill_opacity(1.0), run_time=0.1)
        
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Change the third bit from '1' to '0', highlighting it in red to show the error.
        self.play(self.lecture[2].animate.set_color(COLOR_3))
        
        # Issue 29: Use scale_factor=1.2 and grid pos C4
        new_bit_3 = Text("0", font_size=48, color=COLOR_3)
        self.place_at_grid(new_bit_3, "C4", scale_factor=1.2)
        
        self.play(
            Transform(bits_mobjects[2], new_bit_3),
            FadeOut(noise_wave)
        )
        # L004: Use Indicate
        self.play(Indicate(bits_mobjects[2], color=COLOR_3))
        
        self.wait(2)
