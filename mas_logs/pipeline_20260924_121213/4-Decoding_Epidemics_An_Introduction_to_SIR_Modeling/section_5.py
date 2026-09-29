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
        lecture_lines = ["Models help simulate intervention strategies.", "Social distancing can flatten the curve.", "Models inform vital public health decisions."]
        self.setup_layout("Summary and Real-World Application", lecture_lines)
        
        # Assets
        hospital = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hospital.svg")
        mask = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mask.svg")

        # === Animation for Lecture Line 1 ===
        # Summarize SIR model power in #FFFFFF, displaying hospital asset
        text1 = Text("SIR Model Power:", font_size=32, color="#FFFFFF")
        desc1 = Text("Predicts epidemic trajectory", font_size=24, color="#FFFFFF")
        group1 = VGroup(text1, desc1, hospital).arrange(DOWN)
        self.place_in_area(group1, 'A4', 'B6', scale_factor=0.6) # Grid 4-6 restriction (B002)
        self.play(FadeIn(group1))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Fade in 'Applications: Pandemics, Networks' in #FFFF00.
        text2 = Text("Applications:", font_size=32, color="#FFFF00")
        list2 = Text("Pandemics, Networks", font_size=24, color="#FFFF00")
        group2 = VGroup(text2, list2).arrange(DOWN)
        self.place_in_area(group2, 'C4', 'D6', scale_factor=0.6) # Grid 4-6 restriction (B002)
        self.play(FadeIn(group2))
        self.lecture[1].set_color("#FFFF00")
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Fade out all with final call to action, showing mask asset
        self.lecture[2].set_color("#00FF00")
        final_call = Text("Make Informed Decisions!", font_size=36, color="#00FF00")
        final_group = VGroup(final_call, mask).arrange(DOWN)
        self.place_in_area(final_group, 'E4', 'F6', scale_factor=0.7)
        self.play(FadeOut(group1), FadeOut(group2), FadeIn(final_group))
        self.wait(4)
        
        self.play(FadeOut(final_group))
        self.wait(1)
