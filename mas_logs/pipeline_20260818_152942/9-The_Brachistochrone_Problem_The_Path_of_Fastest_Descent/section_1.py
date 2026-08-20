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

class Section1Scene(TeachingScene):
    def construct(self):
        lines = ["What path gets a marble down fastest?", "A straight line seems the most direct.", "But a curved path can gain speed earlier."]
        self.setup_layout("The Brachistochrone Problem", lines)
        
        # Load asset
        marble_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/marble.svg"
        
        # Define objects
        marble_start = SVGMobject(marble_asset, color=WHITE)
        straight_path = Line(start=self.grid["B2"], end=self.grid["E5"], color="#FFD700")
        curved_path = ArcBetweenPoints(start=self.grid["B2"], end=self.grid["E5"], angle=-TAU/8, color="#00FFFF")
        
        # Grouping for area placement
        animation_group = VGroup(straight_path, curved_path)
        self.place_in_area(animation_group, 'B2', 'E5', scale_factor=1.0)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]))
        self.place_at_grid(marble_start, "B3", scale_factor=0.8)
        self.play(FadeIn(marble_start))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(self.lecture[1]))
        self.play(Create(straight_path))
        self.lecture[1].set_color("#FFD700")
        
        marble_moving = SVGMobject(marble_asset, color="#FFD700")
        self.place_at_grid(marble_moving, "D3", scale_factor=0.8)
        self.play(MoveAlongPath(marble_moving, straight_path), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(self.lecture[2]))
        self.play(Create(curved_path))
        self.lecture[2].set_color("#00FFFF")
        
        marble_curved = SVGMobject(marble_asset, color="#00FFFF")
        self.place_at_grid(marble_curved, "D5", scale_factor=0.8)
        self.play(MoveAlongPath(marble_curved, curved_path), run_time=1.2)
        
        # Final highlight
        self.play(curved_path.animate.set_color("#FF4500"))
        self.wait(2)
