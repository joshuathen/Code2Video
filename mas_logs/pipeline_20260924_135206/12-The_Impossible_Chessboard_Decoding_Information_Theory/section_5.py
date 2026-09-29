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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Conclusion: Information Theory Significance", [
            "Strategy applies Error Correction Code principles.",
            "Information theory minimizes uncertainty via parity.",
            "Technology uses this to fix data corruption."
        ])
        
        phone_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/phone.svg"

        # === Animation for Lecture Line 1 ===
        # Strategy applies Error Correction Code principles.
        # Display a QR code (simulated) next to a phone icon.
        qr_code = Square(side_length=1.5, color=WHITE).set_fill(WHITE, opacity=0.3)
        phone = SVGMobject(phone_path).scale(0.5)
        qr_group = VGroup(qr_code, phone).arrange(RIGHT)
        self.place_in_area(qr_group, 'A3', 'B5', scale_factor=0.6)
        self.play(Create(qr_group))
        self.play(self.lecture[0].animate.set_color(BLUE))

        # === Animation for Lecture Line 2 ===
        # Information theory minimizes uncertainty via parity.
        # Formula for uncertainty/parity bit
        formula = MathTex(r"H = -\\sum p_i \\log p_i", color=TEAL)
        self.place_in_area(formula, 'B3', 'C5', scale_factor=0.9)
        parity_bit = Circle(radius=0.2, color="#00FFFF").set_fill("#00FFFF", opacity=0.8)
        self.place_at_grid(parity_bit, 'A4', scale_factor=0.8)
        self.play(Write(formula), FadeIn(parity_bit))
        self.play(self.lecture[1].animate.set_color("#00FFFF"))

        # === Animation for Lecture Line 3 ===
        # Technology uses this to fix data corruption.
        # Final summary text fading in with a bright white background glow, appearing alongside phone.
        data_block = Rectangle(height=1.0, width=1.5, color=GREEN)
        label = Text("Error-Corrected Data", font_size=16)
        data_group = VGroup(data_block, label).arrange(DOWN)
        phone_2 = SVGMobject(phone_path).scale(0.4)
        final_group = VGroup(data_group, phone_2).arrange(RIGHT)
        self.place_in_area(final_group, 'D3', 'E5', scale_factor=0.7)
        
        glow = BackgroundRectangle(final_group, color=WHITE, fill_opacity=0.2)
        
        self.play(FadeIn(glow), FadeIn(final_group))
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(2)
