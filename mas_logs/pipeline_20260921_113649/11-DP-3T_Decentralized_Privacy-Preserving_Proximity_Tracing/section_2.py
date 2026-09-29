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
        self.setup_layout("Core Mechanism: Ephemeral Tracing Keys", [
            "Phones generate secret Daily Tracing Keys.",
            "Keys derive random Ephemeral IDs.",
            "IDs rotate every fifteen minutes."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Using SVG asset
        phone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/phone.svg")
        daily_key = Dot(color="#00FF00", radius=0.3)
        key_label = Text("Daily Key", font_size=20, color="#00FF00")
        
        daily_key_group = VGroup(phone, daily_key, key_label).arrange(DOWN)
        self.place_at_grid(daily_key_group, 'B4', scale_factor=0.85)
        
        self.play(FadeIn(daily_key_group), Write(key_label))
        self.lecture[0].set_color("#00FF00")

        # === Animation for Lecture Line 2 ===
        eph_key1 = Dot(color="#FFFF00", radius=0.2)
        eph_key2 = Dot(color="#FFFF00", radius=0.2)
        eph_key3 = Dot(color="#FFFF00", radius=0.2)
        ephemeral_keys_group = VGroup(eph_key1, eph_key2, eph_key3).arrange(RIGHT)
        
        self.place_in_area(ephemeral_keys_group, 'C2', 'C4', scale_factor=0.75)
        
        self.play(
            FadeIn(ephemeral_keys_group), 
            daily_key.animate.set_color("#FFFF00") 
        )
        self.lecture[1].set_color("#FFFF00")

        # === Animation for Lecture Line 3 ===
        rotation_group = VGroup(eph_key1, eph_key2, eph_key3)
        self.play(Rotate(rotation_group, angle=PI, about_point=self.grid['D3']))
        self.lecture[2].set_color("#FFFF00")
        self.wait(1)
