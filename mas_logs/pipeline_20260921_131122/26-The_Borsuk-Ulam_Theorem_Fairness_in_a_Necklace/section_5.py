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
        self.setup_layout("Conclusion: Why Symmetry Matters", [
            "Topology guarantees fairness in systems.",
            "Perfect balance is mathematically inevitable.",
            "Symmetry is fundamental to fairness."
        ])
        
        # Visuals - using SVGs as requested by storyboard and critic
        symmetry_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/scale.svg").set_color("#00FFFF")
        balance_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/scale.svg").set_color("#FFFFFF")
        final_summary = VGroup(
            Text("Fairness", font_size=36, color=WHITE),
            Text("Guaranteed by Math", font_size=24, color="#FF00FF")
        ).arrange(DOWN)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        self.place_at_grid(symmetry_icon, 'B2', scale_factor=0.7)
        self.play(FadeIn(symmetry_icon))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFFFF"))
        self.place_at_grid(balance_icon, 'D3', scale_factor=0.8)
        self.play(FadeIn(balance_icon))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF00FF"))
        # Load final asset as requested in storyboard
        final_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/scale.svg").set_color("#FF00FF")
        self.place_at_grid(final_asset, 'F6', scale_factor=0.4)
        self.place_in_area(final_summary, 'E2', 'F6', scale_factor=0.6)
        self.play(FadeIn(final_asset), Write(final_summary))
        self.wait(2)
