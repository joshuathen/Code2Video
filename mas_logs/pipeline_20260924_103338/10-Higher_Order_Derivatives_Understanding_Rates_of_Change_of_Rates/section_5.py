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
        self.setup_layout("Summary & Synthesis", [
            "We now have a full derivative hierarchy.",
            "These define the 'DNA' of the motion.",
            "Use this to predict behavior in complex systems."
        ])
        
        # Assets
        hierarchy_table = VGroup(
            Text("f(x): Position", font_size=20, color="#FFCC00"),
            Text("f'(x): Velocity", font_size=20, color="#00FFCC"),
            Text("f''(x): Acceleration", font_size=20, color="#FF66FF"),
            Text("f'''(x): Jerk", font_size=20, color="#FF9966")
        ).arrange(DOWN, aligned_edge=LEFT)
        
        # Use assets per instructions
        dna_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/dna.svg")
        target_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/target.svg")
        
        app_icons = VGroup(
            Square(side_length=0.5, color="#FFFFFF"),
            Circle(radius=0.25, color="#FFFFFF"),
            target_icon
        ).arrange(RIGHT)

        # === Animation for Lecture Line 1 ===
        self.place_at_grid(hierarchy_table, 'B2', scale_factor=1.0)
        # Adding DNA asset to hierarchy table
        self.place_at_grid(dna_icon, 'B5', scale_factor=0.3)
        self.lecture[0].set_color("#FFCC00")
        self.play(FadeIn(hierarchy_table), FadeIn(dna_icon))
        self.wait(4)

        # === Animation for Lecture Line 2 ===
        # DNA is already displayed above, maybe just highlight it
        self.lecture[1].set_color("#00FFCC")
        self.play(Indicate(dna_icon))
        self.wait(4)

        # === Animation for Lecture Line 3 ===
        self.place_at_grid(app_icons, 'E4', scale_factor=0.9)
        self.lecture[2].set_color("#FF9966")
        self.play(FadeIn(app_icons))
        self.wait(4)

        # Conclusion
        self.play(FadeOut(hierarchy_table), FadeOut(dna_icon), FadeOut(app_icons))
        self.play(FadeOut(self.lecture), FadeOut(self.title))
