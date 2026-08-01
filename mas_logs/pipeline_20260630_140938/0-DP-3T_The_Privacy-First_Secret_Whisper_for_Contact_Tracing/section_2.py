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
        # Setup the layout with section title and lecture lines
        self.setup_layout(
            "Prerequisite Knowledge: Cryptographic Hashes & BLE",
            [
                "Cryptographic hashes turn data into irreversible unique strings.",
                "Bluetooth Low Energy allows devices to exchange short-range signals.",
                "These beacons act as proximity-based digital handshakes between users."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Description: A grey 'Meat Grinder' icon (#888888) takes in 'Secret' text and outputs a white hex string (#FFFFFF).
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Meat Grinder Representation (Grey icon #888888)
        grinder_body = Square(side_length=0.8, color="#888888", fill_opacity=0.5)
        grinder_label = Text("HASH", font_size=16, color="#888888")
        grinder = VGroup(grinder_body, grinder_label)
        self.place_at_grid(grinder, "B3")
        
        input_text = Text("Secret", font_size=20, color=WHITE)
        self.place_at_grid(input_text, "B1")
        
        output_text = Text("f3a2...8b1e", font_size=18, color=WHITE)
        output_text.set_opacity(0)
        self.place_at_grid(output_text, "B5")
        
        self.play(FadeIn(grinder))
        self.play(input_text.animate.move_to(self.grid["B3"]), run_time=1.5)
        self.play(
            FadeOut(input_text, shift=DOWN*0.3),
            FadeIn(output_text, shift=DOWN*0.3)
        )
        self.play(output_text.animate.move_to(self.grid["B5"]), run_time=1.5)
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        # Description: Two phone icons (#0000FF) emit pulsating light-blue ripples (#ADD8E6) towards each other.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(BLUE)
        )
        
        # Using the phone asset from common directory
        phone_path = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/phone.svg"
        phone1 = SVGMobject(phone_path, color="#0000FF", height=0.8)
        phone2 = SVGMobject(phone_path, color="#0000FF", height=0.8)
        
        self.place_at_grid(phone1, "E2")
        self.place_at_grid(phone2, "E5")
        
        self.play(FadeIn(phone1), FadeIn(phone2))
        
        # Ripple effect setup
        def create_ripple(pos):
            ripple = Circle(radius=0.1, color="#ADD8E6", stroke_width=2)
            ripple.move_to(pos)
            return ripple

        # First pulse of ripples
        ripple1 = create_ripple(self.grid["E2"])
        ripple2 = create_ripple(self.grid["E5"])
        self.add(ripple1, ripple2)
        
        self.play(
            ripple1.animate.scale(10).set_stroke(opacity=0),
            ripple2.animate.scale(10).set_stroke(opacity=0),
            run_time=2
        )
        self.remove(ripple1, ripple2)

        # === Animation for Lecture Line 3 ===
        # Description: When ripples meet, a small white handshake icon (#FFFFFF) appears between the phones.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(GREEN)
        )
        
        # Handshake icon (simple white representation #FFFFFF)
        hand1 = Arc(radius=0.2, start_angle=0, angle=PI, color=WHITE)
        hand2 = Arc(radius=0.2, start_angle=PI, angle=PI, color=WHITE)
        hand2.shift(RIGHT*0.1)
        handshake = VGroup(hand1, hand2)
        self.place_in_area(handshake, "E3", "E4", scale_factor=1.2)
        
        # Second pulse meeting in the middle to trigger the handshake
        ripple3 = create_ripple(self.grid["E2"])
        ripple4 = create_ripple(self.grid["E5"])
        self.add(ripple3, ripple4)
        
        self.play(
            ripple3.animate.scale(8).set_stroke(opacity=0),
            ripple4.animate.scale(8).set_stroke(opacity=0),
            FadeIn(handshake, scale=0.5),
            run_time=2
        )
        self.wait(3)
