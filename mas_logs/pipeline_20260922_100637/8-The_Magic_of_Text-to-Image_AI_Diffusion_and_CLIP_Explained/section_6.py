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

class Section6Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Conclusion and Future Outlook", [
            "Math transforms chaos into meaningful art.",
            "From static noise emerges structured beauty.",
            "AI unlocks infinite creative possibilities today."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Math transforms chaos into meaningful art.
        self.lecture[0].set_color(BLUE)
        
        computer_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg")
        list_group = VGroup(
            Text("1. Noise to Latent Space", font_size=20),
            Text("2. Diffusion Process", font_size=20),
            Text("3. CLIP Guidance", font_size=20),
            computer_icon
        ).arrange(DOWN, aligned_edge=LEFT)
        
        self.place_in_area(list_group, "A2", "C4", scale_factor=0.8)
        self.play(Write(list_group))

        # === Animation for Lecture Line 2 ===
        # From static noise emerges structured beauty.
        self.lecture[1].set_color(YELLOW)
        
        # Redoing arrow per fix
        arrow = Arrow(start=self.grid["C3"], end=self.grid["E3"], color=WHITE)
        self.place_at_grid(arrow, "D3", scale_factor=0.6)
        self.play(Create(arrow))

        # === Animation for Lecture Line 3 ===
        # AI unlocks infinite creative possibilities today.
        self.lecture[2].set_color(GREEN)
        
        robot_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        end_msg = VGroup(
            Text("Ready to build?", font_size=32, color=WHITE),
            robot_icon
        ).arrange(DOWN)
        
        self.place_at_grid(end_msg, "E4", scale_factor=1.0)
        self.play(FadeIn(end_msg))
        self.wait(2)
