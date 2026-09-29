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
        self.setup_layout("Step 3: The Match Process", [
            "Positive users upload keys to servers.",
            "Phones download keys to check matches.",
            "Local matching alerts users of exposure."
        ])
        
        # Assets
        server_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/server.svg")
        server_icon.set_color("#88B04B")
        server_label = Text("Match Server", font_size=18)
        server_group = VGroup(server_icon, server_label).arrange(DOWN)
        self.place_at_grid(server_group, 'B3', scale_factor=0.9)
        self.add(server_group)

        bob_phone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/phone.svg")
        bob_label = Text("Bob's Phone", font_size=16)
        bob_group = VGroup(bob_phone, bob_label).arrange(DOWN)
        self.place_at_grid(bob_group, 'D3', scale_factor=1.0)
        self.add(bob_group)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#88B04B")
        dtk = Dot(color="#EFC050")
        dtk_label = Text("DTK", font_size=16).next_to(dtk, UP)
        dtk_group = VGroup(dtk, dtk_label)
        self.play(FadeIn(dtk_group))
        self.play(dtk_group.animate.move_to(server_icon.get_center()))
        self.play(FadeOut(dtk_group))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#6B5B95")
        regenerated_ids = VGroup(*[Dot(color="#6B5B95") for _ in range(3)]).arrange(RIGHT)
        self.place_at_grid(regenerated_ids, 'D4')
        self.play(FadeIn(regenerated_ids))
        self.play(regenerated_ids.animate.move_to(bob_phone.get_center()))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#D64161")
        match_found = Text("Match Found!", color="#D64161", font_size=24, weight=BOLD)
        self.place_at_grid(match_found, 'E3', scale_factor=0.8)
        self.play(FadeIn(match_found))
        self.play(Indicate(match_found))
        self.wait(2)
