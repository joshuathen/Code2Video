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
        lecture_lines = ["Phones broadcast ephemeral identifiers via Bluetooth.", 
                         "Nearby phones record these identifiers locally.", 
                         "History logs remain secure on devices."]
        self.setup_layout("Step 2: Proximity Exchange & Storage", lecture_lines)
        
        # Load assets
        alice = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/alice.svg")
        bob = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bob.svg")
        alice_phone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/phone.svg")
        bob_phone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/phone.svg")

        # Position Alice and Bob
        self.place_at_grid(alice, 'A3', scale_factor=0.4)
        self.place_at_grid(bob, 'A5', scale_factor=0.4)
        self.place_at_grid(alice_phone, 'B3', scale_factor=0.3)
        self.place_at_grid(bob_phone, 'B5', scale_factor=0.3)
        
        self.add(alice, bob, alice_phone, bob_phone)
        self.add(Text("Alice", font_size=18).next_to(alice, UP))
        self.add(Text("Bob", font_size=18).next_to(bob, UP))

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF6F61")
        alice_signal = Circle(radius=0.3, color="#FF6F61", stroke_width=2).move_to(alice_phone)
        bob_signal = Circle(radius=0.3, color="#6B5B95", stroke_width=2).move_to(bob_phone)
        self.play(Create(alice_signal), Create(bob_signal))
        self.play(Flash(alice_signal), Flash(bob_signal))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#88B04B")
        ephid = Text("EphID", font_size=16, color="#88B04B")
        ephid.move_to(alice_phone.get_center())
        self.play(ephid.animate.move_to(bob_phone.get_center()))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#EFC050")
        log = Rectangle(height=0.6, width=1.0, color="#EFC050", fill_opacity=0.3)
        self.place_in_area(log, 'D3', 'D4', scale_factor=0.9)
        log_label = Text("History Log", font_size=14, color="#EFC050")
        self.place_at_grid(log_label, 'E3', scale_factor=0.7)
        
        self.play(Create(log), Write(log_label))
        self.wait(2)
