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
        lecture_lines = [
            "Devices use Bluetooth to broadcast ephemeral identifiers.",
            "Rolling Proximity Identifiers, or RPIs, change periodically.",
            "This hierarchy prevents long-term tracking of users.",
            "The daily key generates unique hourly identifiers.",
            "IDs rotate to maintain user privacy during broadcasts."
        ]
        self.setup_layout("Core Mechanism: Rolling Proximity Identifiers (RPIs)", lecture_lines)
        
        # Assets
        smartphone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/smartphone.svg")
        self.place_at_grid(smartphone, 'B3', scale_factor=1.5)
        
        # Mobjects
        rpi_label = Text("RPI: A5B2-C9F1", font_size=24)
        user_node = Circle(radius=0.3, color=WHITE).set_fill(BLUE, opacity=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FFFF")
        self.add(smartphone)
        self.place_in_area(rpi_label, 'A2', 'A5', scale_factor=0.9)
        self.play(FadeIn(smartphone), Write(rpi_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF00FF")
        rpi_alt = Text("RPI: X921-M3L0", font_size=24)
        self.place_in_area(rpi_alt, 'A2', 'A5', scale_factor=0.9)
        self.play(Transform(rpi_label, rpi_alt))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        self.place_at_grid(user_node, 'E3', scale_factor=0.7)
        # Diagram
        arrow = Arrow(start=rpi_label.get_bottom(), end=user_node.get_top(), color="#FFFF00")
        self.play(FadeIn(user_node), Create(arrow))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#00FF00")
        key_label = Text("Daily Key", font_size=24, color="#00FF00")
        self.place_at_grid(key_label, 'C3', scale_factor=0.8)
        self.play(Write(key_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FF0000")
        # Final rotation visual
        self.play(Rotate(rpi_label, angle=2*PI, run_time=1.5))
        self.wait(2)
