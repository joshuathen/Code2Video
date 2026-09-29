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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Phones exchange IDs via Bluetooth.", "No location data is stored.", "Only encountered IDs are recorded."]
        self.setup_layout("The Handshake: Exchanging Ephemeral IDs", lecture_lines)
        
        # Animation Elements
        alice_phone = Rectangle(width=0.8, height=1.4, color=BLUE).add(Text("Alice", font_size=12).shift(UP*0.3))
        bob_phone = Rectangle(width=0.8, height=1.4, color=GREEN).add(Text("Bob", font_size=12).shift(UP*0.3))
        
        # Repositioned per instructions
        self.place_at_grid(alice_phone, "C2", scale_factor=0.7)
        self.place_at_grid(bob_phone, "C5", scale_factor=0.7)
        
        id_packet = Text("ID: #A1-001", font_size=18, color=YELLOW)
        # Repositioned per instructions
        self.place_in_area(id_packet, "B3", "B3", scale_factor=0.6)
        id_packet.move_to(alice_phone.get_right() + RIGHT * 0.5)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.play(FadeIn(alice_phone), FadeIn(bob_phone))
        self.play(id_packet.animate.move_to(bob_phone.get_left() + LEFT * 0.5), run_time=1.5)
        self.play(FadeOut(id_packet))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(GREEN)
        loc_icon = Cross(color=RED).scale(0.5).next_to(alice_phone, DOWN)
        self.play(Create(loc_icon))
        self.wait(1)
        self.play(FadeOut(loc_icon))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        memory_box = Rectangle(width=1.2, height=1.0, color=WHITE).next_to(bob_phone, RIGHT, buff=0.5)
        memory_text = Text("Mem: #A1-001", font_size=14).move_to(memory_box)
        self.play(Create(memory_box), Write(memory_text))
        self.wait(2)
