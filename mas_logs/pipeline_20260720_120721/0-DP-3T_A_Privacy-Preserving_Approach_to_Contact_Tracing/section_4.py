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
        lecture_lines = [
            "Your phone checks against reported keys.",
            "It alerts you if you were exposed.",
            "Privacy is maintained throughout the process."
        ]
        self.setup_layout("How It Works: Reporting and Infection Status", lecture_lines)
        
        # Define colors for lecture lines and corresponding animations
        color1 = "#FFD700"  # Gold
        color2 = "#87CEEB"  # Sky Blue
        color3 = "#98FB98"  # Pale Green

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(color1))
        
        reporting_exposure_text = Text("Reporting Exposure", font_size=20, color=color1)
        phone_icon_1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/phone.svg").set_color(color1).scale(0.4)
        
        reporting_exposure_group = VGroup(reporting_exposure_text, phone_icon_1).arrange(RIGHT, buff=0.5)
        self.place_at_grid(reporting_exposure_group, 'C2', scale_factor=0.7)
        self.play(FadeIn(reporting_exposure_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(color2))
        
        infection_status_update_text = Text("Infection Status Update", font_size=20, color=color2)
        phone_icon_2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/phone.svg").set_color(color2).scale(0.4)
        
        infection_status_update_group = VGroup(infection_status_update_text, phone_icon_2).arrange(RIGHT, buff=0.5)
        self.place_at_grid(infection_status_update_group, 'D2', scale_factor=0.7)
        self.play(FadeIn(infection_status_update_group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(color3))
        
        anonymized_data_sharing_text = Text("Anonymized Data Sharing", font_size=20, color=color3)
        phone_icon_3 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/phone.svg").set_color(color3).scale(0.4)
        
        anonymized_data_sharing_group = VGroup(anonymized_data_sharing_text, phone_icon_3).arrange(RIGHT, buff=0.5)
        self.place_at_grid(anonymized_data_sharing_group, 'E2', scale_factor=0.7)
        self.play(FadeIn(anonymized_data_sharing_group))
        self.wait(2)

        self.play(
            FadeOut(reporting_exposure_group), 
            FadeOut(infection_status_update_group), 
            FadeOut(anonymized_data_sharing_group)
        )
        self.wait(1)
