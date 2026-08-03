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
        # Initialize Layout
        self.setup_layout("Prerequisite: Binary Counting as a Pulse", [
            "Solve this puzzle using the heartbeat of binary counting.",
            "Watch bits flip as we count zero to seven.",
            "The rightmost changing bit tells which disk to move."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Solve this puzzle using the heartbeat of binary counting.
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Watch bits flip as we count zero to seven.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Binary strings 001 to 111
        binary_strings = ["001", "010", "011", "100", "101", "110", "111"]
        binary_mobjects = []
        binary_vgroup = VGroup()
        
        for s in binary_strings:
            # Create bit-by-bit group to allow individual highlighting
            # Use Monospace for aligned columns
            bits = VGroup(*[Text(bit, font="Monospace", font_size=28, color="#00FFFF") for bit in s])
            bits.arrange(RIGHT, buff=0.15)
            binary_mobjects.append(bits)
            binary_vgroup.add(bits)
            
        binary_vgroup.arrange(DOWN, buff=0.3)
        
        # [RESOLVE ISSUE 29] Position the vertical list on the right side area
        self.place_in_area(binary_vgroup, 'A4', 'F4', scale_factor=1.1)
        
        # Fade in the binary list one by one
        for bits in binary_mobjects:
            self.play(FadeIn(bits), run_time=0.3)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The rightmost changing bit tells which disk to move.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Pulse animation logic for the rightmost bit that is '1'
        # Sequence of indices for rightmost 1 in each binary string:
        # 001(2), 010(1), 011(2), 100(0), 101(2), 110(1), 111(2)
        indices = [2, 1, 2, 0, 2, 1, 2]
        
        # Pulse indicator (not added to scene yet)
        pulse_circle = Circle(radius=0.2, color="#FFFF00").set_stroke(width=3)
        
        for i, bits in enumerate(binary_mobjects):
            target_idx = indices[i]
            target_bit = bits[target_idx]
            
            # Highlight and Pulse
            # We use a copy of pulse_circle for the animation
            temp_pulse = pulse_circle.copy().move_to(target_bit.get_center())
            self.add(temp_pulse)
            
            self.play(
                target_bit.animate.set_color("#FFFF00"),
                temp_pulse.animate.scale(2.5).set_stroke(opacity=0),
                Flash(target_bit, color="#FFFF00", flash_radius=0.25, line_length=0.1, num_lines=8),
                run_time=0.6
            )
            self.remove(temp_pulse)
            
            # Brief pause between highlights
            self.wait(0.1)

        self.wait(2)
