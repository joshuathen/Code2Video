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
        lecture_lines = [
            "Focus on the ellipse intersection.",
            "Sum of distances remains constant.",
            "Tangents from point to sphere are equal.",
            "This proves the ellipse property.",
            "Geometry made elegantly visible."
        ]
        self.setup_layout("The Proof Logic (Focus on Ellipse)", lecture_lines)
        
        # Ellipse and Points Setup
        ellipse = Ellipse(width=3.0, height=1.5, color=GREEN)
        # Positioning per feedback (Issue 27, 38)
        self.place_in_area(ellipse, 'C3', 'E5', scale_factor=0.9)
        
        # Load asset
        sphere_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg")
        sphere_asset.set_color("#00FFFF")
        
        # Foci Positioning per feedback (Issue 28, 38)
        f1 = sphere_asset.copy()
        self.place_at_grid(f1, 'D2', scale_factor=0.3)
        
        f2 = sphere_asset.copy()
        self.place_at_grid(f2, 'D5', scale_factor=0.3)
        
        f1_label = Text("F1", font_size=18, color="#00FFFF").next_to(f1, UP, buff=0.1)
        f2_label = Text("F2", font_size=18, color="#00FFFF").next_to(f2, UP, buff=0.1)
        
        p = Dot(color=WHITE)
        p.move_to(ellipse.point_from_proportion(0))
        
        line1 = Line(p.get_center(), f1.get_center(), color=ORANGE)
        line2 = Line(p.get_center(), f2.get_center(), color=ORANGE)
        
        # ValueTracker for robust updates
        t_tracker = ValueTracker(0)
        
        def update_p(mob):
            t = t_tracker.get_value()
            mob.move_to(ellipse.point_from_proportion(t % 1))
            
        def update_l1(mob):
            mob.put_start_and_end_on(p.get_center(), f1.get_center())
            
        def update_l2(mob):
            mob.put_start_and_end_on(p.get_center(), f2.get_center())
            
        p.add_updater(update_p)
        line1.add_updater(update_l1)
        line2.add_updater(update_l2)
        
        # === Animation for Lecture Line 1 ===
        self.play(Create(ellipse), run_time=1)
        self.lecture[0].set_color(GREEN)
        
        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(f1), FadeIn(f2), Write(f1_label), Write(f2_label), run_time=1)
        self.lecture[1].set_color("#00FFFF")
        
        # === Animation for Lecture Line 3 ===
        self.play(Create(p), Create(line1), Create(line2), run_time=1)
        self.lecture[2].set_color(ORANGE)
        
        # === Animation for Lecture Line 4 ===
        self.play(t_tracker.animate.set_value(1), run_time=2, rate_func=linear)
        self.lecture[3].set_color("#00FFFF")
        
        # === Animation for Lecture Line 5 ===
        self.wait(1)
        self.lecture[4].set_color(WHITE)
        
        # Clean up
        p.remove_updater(update_p)
        line1.remove_updater(update_l1)
        line2.remove_updater(update_l2)
