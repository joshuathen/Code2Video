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
        lecture_lines = ["Phones whisper IDs via Bluetooth.", "Nearby devices store these IDs locally.", "No GPS data is ever recorded."]
        self.setup_layout("The Proximity Exchange Mechanism", lecture_lines)
        
        phone_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/phone.svg"
        phone_a = SVGMobject(phone_path, color=WHITE)
        phone_b = SVGMobject(phone_path, color=WHITE)
        
        # Applying grid positioning as requested by feedback
        self.place_at_grid(phone_a, 'C1', scale_factor=0.8)
        self.place_at_grid(phone_b, 'C6', scale_factor=0.8)
        
        id_packet = Dot(color=YELLOW)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(FadeIn(phone_a), FadeIn(phone_b))
        
        # Display ephemeral ID packets moving between the devices
        id_packet.move_to(phone_a.get_right())
        self.play(id_packet.animate.move_to(phone_b.get_left()), run_time=1.5)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        signal_animation = Circle(radius=0.3, color=GREEN)
        self.place_at_grid(signal_animation, 'D5', scale_factor=0.6)
        self.play(FadeIn(signal_animation))
        self.play(Flash(phone_b, color=GREEN))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(RED))
        
        gps_symbol = Cross(color=RED)
        gps_text = Text("No GPS", font_size=20, color=RED)
        no_gps_group = VGroup(gps_symbol, gps_text).arrange(RIGHT)
        self.place_in_area(no_gps_group, 'A3', 'A4', scale_factor=0.7)
        
        self.play(FadeIn(no_gps_group))
        self.wait(2)
