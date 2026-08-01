from manim import *
import numpy as np

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
        self.setup_layout(
            "The Digital Handshake (Log Storage)", 
            [
                "Nearby phones exchange rolling IDs via Bluetooth.", 
                "Observed codes are stored in a local, encrypted diary.", 
                "No location data or identities are ever recorded."
            ]
        )

        # --- Assets ---
        phone_asset_path = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/phone.svg"

        def create_phone_mobject(label_text, color=WHITE):
            phone_svg = SVGMobject(phone_asset_path).set_color(color).scale(0.4)
            label = Text(label_text, font_size=16, color=color).next_to(phone_svg, UP, buff=0.1)
            return VGroup(phone_svg, label)

        pip_phone = create_phone_mobject("Pip", WHITE)
        leo_phone = create_phone_mobject("Leo", WHITE)

        # Bluetooth waves (pulsing circles)
        def create_waves(anchor, color="#ADD8E6"):
            waves = VGroup(*[
                Circle(radius=0.1 + i * 0.2, stroke_width=2, color=color, stroke_opacity=1 - i*0.3)
                for i in range(3)
            ])
            waves.add_updater(lambda m: m.move_to(anchor.get_center()))
            return waves

        pip_waves = create_waves(pip_phone[0]) # anchor to svg part
        leo_waves = create_waves(leo_phone[0])

        # Local Diary Folder
        folder_rect = Rectangle(width=0.6, height=0.4, fill_opacity=0.3, color="#90EE90")
        folder_tab = Polygon([0,0,0], [0.2,0,0], [0.15,0.1,0], [0,0.1,0], color="#90EE90").shift(UP*0.2 + LEFT*0.2)
        diary_folder = VGroup(folder_rect, folder_tab)
        diary_label = Text("Local Diary", font_size=12, color="#90EE90").next_to(diary_folder, DOWN, buff=0.1)
        diary_group = VGroup(diary_folder, diary_label)

        # Rolling ID
        code_id = Text("A12B", font_size=18, color=WHITE)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#ADD8E6")
        
        # Initial positions to allow passing motion
        self.place_at_grid(pip_phone, "B3", scale_factor=1.0)
        self.place_at_grid(leo_phone, "E6", scale_factor=1.0)
        
        self.add(pip_phone, leo_phone)
        
        # Pulse animation logic
        def pulse_anim(m, dt):
            for circle in m:
                circle.scale(1 + 0.5 * dt)
                if circle.width > 1.2:
                    circle.scale(0.2 / circle.radius)
        
        pip_waves.add_updater(pulse_anim)
        leo_waves.add_updater(pulse_anim)
        
        self.play(
            # Move Pip to B5 (Issue 41 Fix)
            pip_phone.animate.move_to(self.grid["B5"]),
            # Move Leo to E3 (Transition)
            leo_phone.animate.move_to(self.grid["E3"]),
            FadeIn(pip_waves),
            FadeIn(leo_waves),
            run_time=4
        )
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#90EE90")
        
        # Stop waves
        pip_waves.clear_updaters()
        leo_waves.clear_updaters()
        
        # Move Leo to E5 and place Diary at E4 (Issue 42 Fixes)
        self.place_at_grid(diary_group, "E4", scale_factor=0.8)
        
        self.play(
            leo_phone.animate.move_to(self.grid["E5"]),
            FadeOut(pip_waves),
            FadeOut(leo_waves),
            FadeIn(diary_group),
            run_time=1.5
        )
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(WHITE) # Using white for the primary text
        
        # Code ID moves from Pip's phone (at B5) to Leo's diary (at E4)
        code_id.move_to(pip_phone.get_center())
        
        self.play(
            code_id.animate.move_to(diary_group.get_center()).scale(0.6),
            run_time=2
        )
        self.play(
            FadeOut(code_id),
            diary_group.animate.scale(1.1).set_color(YELLOW),
            run_time=0.4
        )
        self.play(
            diary_group.animate.scale(1/1.1).set_color("#90EE90"),
            run_time=0.4
        )
        
        # No Location visualization (Issue 43 Fix: F5)
        no_gps_text = Text("GPS", font_size=18, color=RED)
        cross_line1 = Line(LEFT*0.3, RIGHT*0.3, color=RED).rotate(45).move_to(no_gps_text)
        cross_line2 = Line(LEFT*0.3, RIGHT*0.3, color=RED).rotate(-45).move_to(no_gps_text)
        no_gps = VGroup(no_gps_text, cross_line1, cross_line2)
        
        self.place_at_grid(no_gps, "F5")
        self.play(FadeIn(no_gps))
        
        self.wait(2)
