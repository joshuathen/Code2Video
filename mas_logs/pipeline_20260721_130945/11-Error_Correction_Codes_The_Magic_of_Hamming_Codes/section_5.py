from manim import *

# Base class provided by the user
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
        # Data from storyboard
        title_text = "Step-by-Step: Correcting a Bit Flip"
        lecture_lines = [
            "When data arrives, we recalculate all three parity bits.",
            "These results form a binary number called the syndrome.",
            "If the syndrome is zero, the data is error-free.",
            "A non-zero syndrome points directly to the corrupted bit.",
            "Flipping that bit back recovers the original data perfectly."
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # Colors
        COLOR_RECALC = "#00FFFF" # Cyan
        COLOR_SYNDROME = "#FFFFFF" # White
        COLOR_ZERO = "#00FF00" # Green
        COLOR_NONZERO = "#FF0000" # Red
        COLOR_CORRECT = "#00FF00" # Green

        # Bits: Using 1 1 1 1 1 0 1 to ensure syndrome 6 (consistent with outline's correction of bit 6)
        # Note: Outline says 'Received: 1110101' but also says syndrome is 6 (2+4).
        # To get 6 (110), P4 and P2 must be 1. 
        # In 1 1 1 1 1 0 1:
        # P1(1,3,5,7): 1,1,1,1 -> 0
        # P2(2,3,6,7): 1,1,0,1 -> 1
        # P4(4,5,6,7): 1,1,0,1 -> 1
        # Result: 110 (6).
        bits_str = ["1", "1", "1", "1", "1", "0", "1"]
        bit_mobjects = VGroup(*[Text(b, font_size=36) for b in bits_str]).arrange(RIGHT, buff=0.4)
        # Area-Positioning Rule (L003)
        self.place_in_area(bit_mobjects, "B2", "B5", scale_factor=1.0)
        
        # Position labels (L015) - positioned in row parallel to movement/bits
        labels = VGroup(*[Text(str(i+1), font_size=18, color=GRAY) for i in range(7)])
        for i, label in enumerate(labels):
            label.next_to(bit_mobjects[i], UP, buff=0.2).scale(0.8)

        # === Animation for Lecture Line 1 ===
        # "When data arrives, we recalculate all three parity bits."
        self.play(self.lecture[0].animate.set_color(COLOR_RECALC))
        self.play(FadeIn(bit_mobjects), FadeIn(labels))
        
        # Recalculate parity groups visual (L004: Indicate)
        p1_rect = SurroundingRectangle(VGroup(bit_mobjects[0], bit_mobjects[2], bit_mobjects[4], bit_mobjects[6]), color=COLOR_RECALC, buff=0.1)
        p2_rect = SurroundingRectangle(VGroup(bit_mobjects[1], bit_mobjects[2], bit_mobjects[5], bit_mobjects[6]), color=COLOR_RECALC, buff=0.1)
        p4_rect = SurroundingRectangle(VGroup(bit_mobjects[3], bit_mobjects[4], bit_mobjects[5], bit_mobjects[6]), color=COLOR_RECALC, buff=0.1)
        
        self.play(Create(p1_rect))
        self.wait(0.4)
        self.play(ReplacementTransform(p1_rect, p2_rect))
        self.wait(0.4)
        self.play(ReplacementTransform(p2_rect, p4_rect))
        self.wait(0.4)
        self.play(FadeOut(p4_rect))

        # === Animation for Lecture Line 2 ===
        # "These results form a binary number called the syndrome."
        self.play(
            self.lecture[0].animate.set_color(WHITE), 
            self.lecture[1].animate.set_color(COLOR_SYNDROME)
        )
        
        synd_vals = VGroup(
            Text("P4 = 1", font_size=22),
            Text("P2 = 1", font_size=22),
            Text("P1 = 0", font_size=22)
        ).arrange(RIGHT, buff=0.4)
        # Fix Issue 39: Place synd_vals in C3-C5
        self.place_in_area(synd_vals, "C3", "C5", scale_factor=1.0)
        
        syndrome_bin = Text("Syndrome = 110", font_size=30, color=COLOR_SYNDROME)
        self.place_at_grid(syndrome_bin, "D4", scale_factor=1.0)
        
        self.play(Write(synd_vals))
        self.play(Write(syndrome_bin))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "If the syndrome is zero, the data is error-free."
        self.play(
            self.lecture[1].animate.set_color(WHITE), 
            self.lecture[2].animate.set_color(COLOR_ZERO)
        )
        
        zero_info = Text("If 000 → No Error", font_size=24, color=COLOR_ZERO)
        # Fix Issue 39: Place zero_info in C3-C5
        self.place_in_area(zero_info, "C3", "C5", scale_factor=1.0)
        
        # Brief demonstration of the clean case
        self.play(FadeIn(zero_info), synd_vals.animate.set_fill_opacity(0.2))
        self.wait(1)
        self.play(FadeOut(zero_info), synd_vals.animate.set_fill_opacity(1.0))

        # === Animation for Lecture Line 4 ===
        # "A non-zero syndrome points directly to the corrupted bit."
        self.play(
            self.lecture[2].animate.set_color(WHITE), 
            self.lecture[3].animate.set_color(COLOR_NONZERO)
        )
        
        calc_text = Text("110 (binary) = 6", font_size=30, color=COLOR_NONZERO)
        # Fix Issue 37: Place calc_text in E3-E5
        self.place_in_area(calc_text, "E3", "E5", scale_factor=1.0)
        
        # Highlight bit at position 6 (bit_mobjects[5])
        box_6 = SurroundingRectangle(bit_mobjects[5], color=COLOR_NONZERO, buff=0.1)
        
        self.play(Write(calc_text))
        self.play(Create(box_6))
        self.play(Indicate(bit_mobjects[5], color=COLOR_NONZERO))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "Flipping that bit back recovers the original data perfectly."
        self.play(
            self.lecture[3].animate.set_color(WHITE), 
            self.lecture[4].animate.set_color(COLOR_CORRECT)
        )
        
        # Morph 0 to 1 at bit position 6
        corrected_bit = Text("1", font_size=36, color=COLOR_CORRECT)
        corrected_bit.move_to(bit_mobjects[5].get_center())
        
        success = Text("Error Corrected!", font_size=28, color=COLOR_CORRECT)
        # Fix Issue 38: Place success message in F3-F5
        self.place_in_area(success, "F3", "F5", scale_factor=1.0)
        
        self.play(
            ReplacementTransform(bit_mobjects[5], corrected_bit),
            box_6.animate.set_color(COLOR_CORRECT),
            FadeOut(synd_vals),
            FadeOut(syndrome_bin),
            FadeOut(calc_text)
        )
        
        self.play(Write(success))
        self.wait(2)
